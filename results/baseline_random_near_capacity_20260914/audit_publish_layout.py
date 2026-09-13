#!/usr/bin/env python3
"""Read-only publication checks; write one JSON report, never stage or publish.

--pre-cleanup skips only the two-results-directory restriction. All other checks
still apply. Uses installed markdown-it-py; never reads credential stores or
follows repository symlinks into external files.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True
from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEEP_RESULTS = {'baseline_ab128_32_ratio12_20260912', 'baseline_random_near_capacity_20260914'}
LIMIT = 100*2**20
# Findings contain paths only; never print matched credential bytes or snippets.
TOKEN_RE = re.compile(rb'(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{20,255})(?![A-Za-z0-9_])')
PARSER = MarkdownIt('commonmark')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git','--no-pager',*args], cwd=ROOT)


def paths_from_git(*args):
    return {os.fsdecode(p) for p in git(*args).split(b'\0') if p}


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name,value in attrs if name.lower() in ('href','src') and value)


def markdown_links(text):
    output = []
    def walk(tokens, line=1):
        for token in tokens:
            here = token.map[0]+1 if token.map else line
            if token.type in ('link_open','image'):
                value = token.attrGet('href' if token.type=='link_open' else 'src')
                if value: output.append((value,here))
            if token.type in ('html_inline','html_block'):
                parser = HtmlLinks(); parser.feed(token.content)
                output.extend((value,here) for value in parser.links)
            if token.children: walk(token.children,here)
    walk(PARSER.parse(text))
    return output


def local_target(source, href):
    value = unescape(href.strip())
    if not value or value.startswith(('#','//')): return None
    parsed = urlsplit(value)
    if parsed.scheme: return None
    if not parsed.path: return None
    decoded = unquote(parsed.path)
    if decoded.startswith('~/'):
        return {'outside_repository':True}
    path = Path(decoded)
    if path.is_absolute():
        # Absolute workspace links are checked locally; slash-root links use
        # repository-root paths without accessing system directories.
        candidate = path if path.is_relative_to(ROOT) else ROOT/decoded.lstrip('/')
    else:
        candidate = source.parent/path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(ROOT): return {'outside_repository':True}
    return {'path':resolved, 'relative':str(resolved.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-cleanup', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/('publish_layout_pre_cleanup.json' if args.pre_cleanup else 'publish_layout_audit.json')
    output = output.resolve()
    if not output.is_relative_to(ROOT): parser.error('--output must remain inside this repository')
    started = datetime.now(timezone.utc).isoformat()
    assert Path(os.fsdecode(git('rev-parse','--show-toplevel')).strip()).resolve()==ROOT
    tests_path = HERE/'retained_tests_check.json'; tests = json.loads(tests_path.read_text())
    test_output = HERE/'retained_tests_output.txt'
    images_path = HERE/'saturday_images_before_cleanup.json'; image_hashes = json.loads(images_path.read_text())
    cleanup = json.loads((HERE/'cleanup_plan.json').read_text())
    expected_hashes = dict(tests['source_sha256'])
    assert not set(expected_hashes)&set(image_hashes)
    expected_hashes.update(image_hashes)
    expected_hashes[str(test_output.relative_to(ROOT))] = tests['output_sha256']
    visible = paths_from_git('ls-files','--cached','--others','--exclude-standard','-z')
    head_paths = paths_from_git('ls-tree','-r','--name-only','-z','HEAD')
    changed = paths_from_git('diff','--no-ext-diff','--name-only','-z','HEAD')
    checked_files = []; skipped_missing = []; unreadable = []; symlinks = []; token_hits = []
    large = []; observed_hashes = {}; total_bytes = 0; changed_during_scan = []
    for relative in sorted(visible):
        path = ROOT/relative
        try: before = path.lstat()
        except FileNotFoundError:
            skipped_missing.append(relative); continue
        if stat.S_ISLNK(before.st_mode):
            # A Git symlink stores this path string, not its target's contents.
            symlinks.append(relative)
            payload = os.fsencode(os.readlink(path))
            if TOKEN_RE.search(payload): token_hits.append(relative)
            total_bytes += len(payload); checked_files.append(relative)
            continue
        if not stat.S_ISREG(before.st_mode): continue
        checked_files.append(relative); total_bytes += before.st_size
        if before.st_size > LIMIT: large.append({'path':relative,'bytes':before.st_size})
        digest = hashlib.sha256() if relative in expected_hashes else None
        tail = b''; found = False
        try:
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*2**20), b''):
                    if digest: digest.update(block)
                    if not found and TOKEN_RE.search(tail+block): found = True
                    tail = block[-512:]
            after = path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): changed_during_scan.append(relative)
            if digest: observed_hashes[relative] = digest.hexdigest()
            if found: token_hits.append(relative)
        except OSError:
            unreadable.append(relative)
    # Required local preservation files are checked even if an ignore rule
    # accidentally removes them from publication visibility.
    for relative in expected_hashes:
        if relative not in observed_hashes:
            path = ROOT/relative
            if path.is_file() and not path.is_symlink(): observed_hashes[relative] = sha(path)
    hash_failures = [p for p,digest in expected_hashes.items() if observed_hashes.get(p)!=digest]
    unpublished_required = [p for p in expected_hashes if p not in visible]
    test_pass_matches = re.findall(r'\b(\d+) passed\b',test_output.read_text()) if test_output.exists() else []
    tests_ok = (tests['returncode']==0 and len(tests['source_sha256'])==51 and len(tests['tests'])==16
        and '159' in test_pass_matches and not any(p in hash_failures for p in [*tests['source_sha256'],str(test_output.relative_to(ROOT))]))
    images_ok = bool(image_hashes) and not any(p in hash_failures for p in image_hashes)
    actual_dirs = {p.name for p in (ROOT/'results').iterdir() if p.is_dir()}
    dirs_ok = actual_dirs==KEEP_RESULTS

    main_entries = {'README.md','CURRENT_PROJECT_MANIFEST.md','results/README.md',str((HERE/'README.md').relative_to(ROOT))}
    obsolete = [Path('results')/p for p in cleanup['remove_results_dirs']]
    obsolete += [Path(p) for p in cleanup['remove_other_obsolete_paths']]
    def old_area(relative): return any(Path(relative)==p or Path(relative).is_relative_to(p) for p in obsolete)
    markdowns = sorted(p for p in visible if p.lower().endswith('.md') and (ROOT/p).is_file()
        and not (ROOT/p).is_symlink() and not old_area(p))
    broken = []; unpublished_links = []; external_local = []; historical = []; scope = []; link_count = 0
    for relative in markdowns:
        source = ROOT/relative
        full = relative in main_entries or source.is_relative_to(HERE) or relative not in head_paths
        if not full and relative not in changed: continue
        text = source.read_text(encoding='utf-8')
        links = markdown_links(text)
        previous = set()
        if not full:
            previous = {href for href,_ in markdown_links(git('show',f'HEAD:{relative}').decode('utf-8'))}
        scope.append({'path':relative,'mode':'all_current_links' if full else 'introduced_links_only'})
        for href,line in links:
            target = local_target(source,href)
            if target is None: continue
            is_historical = not full and href in previous
            if target.get('outside_repository'):
                if not is_historical: external_local.append({'source':relative,'line_hint':line})
                continue
            dest = target['path']; dest_rel = target['relative']
            row = {'source':relative,'line_hint':line,'target_path':dest_rel}
            exists = dest.exists()
            if is_historical:
                if not exists: historical.append(row)
                continue
            link_count += 1
            if not exists: broken.append(row); continue
            published = dest_rel in visible if dest.is_file() else any(p.startswith(dest_rel.rstrip('/')+'/') for p in visible)
            if not published: unpublished_links.append(row)
    missing_entries = sorted(p for p in main_entries if not (ROOT/p).is_file() or p not in visible)
    checks = {'results_only_two_studies':None if args.pre_cleanup else dirs_ok,
        'retained_159_tests_and_51_source_hashes':tests_ok,'saturday_image_hashes_preserved':images_ok,
        'required_preservation_files_git_visible':not unpublished_required,'main_entries_present':not missing_entries,
        'new_and_entry_markdown_local_targets_exist':not broken and not external_local,
        'markdown_targets_git_visible':not unpublished_links,'git_visible_files_within_100MiB':not large,
        'no_github_token_pattern_paths':not token_hits,'all_visible_regular_files_read':not unreadable,
        'files_stable_during_scan':not changed_during_scan}
    passed = all(v is not False for v in checks.values())
    result = {'mode':'pre_cleanup' if args.pre_cleanup else 'final_layout','started_utc':started,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'passed':passed,
        'ready_for_publish':passed and not args.pre_cleanup,'no_cleanup_stage_commit_or_push':True,
        'scope':'Actual existing paths from git ls-files --cached --others --exclude-standard. Ignored untracked originals are excluded; tracked files remain included even if an ignore rule matches.',
        'checks':checks,'results_directories':sorted(actual_dirs),'expected_results_directories':sorted(KEEP_RESULTS),
        'directory_restriction_skipped':args.pre_cleanup,'retained_test_modules':len(tests['tests']),
        'retained_source_and_test_files':len(tests['source_sha256']),'prior_passed_tests':159,
        'saturday_preserved_images':len(image_hashes),'hash_mismatch_or_missing_paths':hash_failures,
        'required_not_git_visible_paths':unpublished_required,'git_visible_existing_file_count':len(checked_files),
        'git_visible_total_bytes':total_bytes,'git_visible_paths_sha256':hashlib.sha256('\0'.join(sorted(visible)).encode()).hexdigest(),
        'large_files_over_100MiB':large,'github_token_pattern_hit_paths':sorted(set(token_hits)),
        'credential_scan':'Only repository Git-visible bytes and symlink path strings. No match values/snippets, external targets, credential stores, gh config, or Git history objects are searched.',
        'unreadable_paths':unreadable,'changed_during_scan_paths':changed_during_scan,'symlinks_not_followed':symlinks,
        'git_listed_but_absent_paths_skipped':skipped_missing,'markdown_scope':scope,'checked_local_link_count':link_count,
        'missing_main_entry_paths':missing_entries,'broken_local_links':broken,'local_links_outside_repository':external_local,
        'local_targets_not_published':unpublished_links,'preexisting_missing_links_not_new_errors':historical,
        'markdown_rule':'Full checks for main entries and new-study/new Git Markdown. Other modified historical Markdown checks only newly introduced href targets; unchanged historical missing targets are reported but do not fail. Obsolete study directories planned for deletion are not recursively audited.',
        'fragment_scope':'File/directory targets checked; heading fragments are not validated.',
        'source_sha256':{'retained_tests_check.json':sha(tests_path),'saturday_images_before_cleanup.json':sha(images_path),
            'cleanup_plan.json':sha(HERE/'cleanup_plan.json')},'builder_sha256':sha(Path(__file__))}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Intentionally output counts and a report path, never credential contents.
    print(json.dumps({'mode':result['mode'],'passed':passed,'files':len(checked_files),'large_files':len(large),
        'token_hit_paths':len(set(token_hits)),'broken_new_links':len(broken),'unpublished_link_targets':len(unpublished_links),
        'historical_missing_links':len(historical),'output':str(output.relative_to(ROOT))},ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
