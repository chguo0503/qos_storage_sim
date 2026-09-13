#!/usr/bin/env python3
"""Execute the reviewed project cleanup only after all research jobs finish."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_info(path):
    files = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
    return {'path':str(path.relative_to(ROOT)), 'files':len(files),
            'bytes':sum(p.stat().st_size for p in files)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = read(HERE/'cleanup_plan.json')
    frozen = read(HERE/'study_plan.json')['core_hashes']
    tests = read(HERE/'retained_tests_check.json')
    assert tests['returncode'] == 0
    for name, expected in tests['source_sha256'].items():
        assert sha(ROOT/name) == expected, f'Retained file changed: {name}'
    for name in plan['remove_root_py']:
        assert name in frozen and sha(ROOT/name) == frozen[name], f'Obsolete source changed: {name}'
    assert not set(plan['keep_root_files']) & set(plan['remove_root_py'])
    pending = []
    for base in [HERE/'runs', HERE/'long_horizon', HERE/'replays']:
        for command in base.rglob('command.json'):
            if 'failed_attempts' in command.parts:
                continue
            if read(command).get('status') != 'complete':
                pending.append(str(command.relative_to(ROOT)))
    acceptances = {}
    for file in sorted(HERE.glob('audit_remote*acceptance.json')):
        value = read(file)
        assert value.get('passed') is True, file
        acceptances[file.name] = sha(file)
    # The final aggregate audit is produced only after all remote queues and
    # source checks are finished; a partial acceptance list is insufficient.
    final_path = HERE/'final_research_audit.json'
    if args.execute:
        assert not pending, pending
        assert final_path.exists() and read(final_path)['passed']
        final = read(final_path)
        assert acceptances == final['remote_acceptance_sha256']
        assert sha(HERE/'comparison.json') == final['comparison_sha256']
        assert sha(HERE/'key_results.json') == final['key_results_sha256']
        assert sha(HERE/'expected_scientific_cases.json') == final['expected_case_registry_sha256']
        for name, expected_sha in final['supporting_audit_sha256'].items():
            assert sha(HERE/name) == expected_sha
        for row in final['records']:
            case = HERE/row['case']
            assert sha(case/'command.json') == row['command_sha256']
            assert sha(case/row['analysis_file']) == row['analysis_sha256']
    image_path = HERE/'saturday_images_before_cleanup.json'
    if not image_path.exists():
        initial = read(Path('/tmp/qos_before_random_near_capacity_images.json'))
        subset = {p:h for p,h in initial.items() if p.startswith('results/baseline_ab128_32_ratio12_20260912/')}
        assert subset
        image_path.write_text(json.dumps(subset,ensure_ascii=False,indent=2)+'\n')
    image_hashes = read(image_path)
    assert all((ROOT/name).exists() and sha(ROOT/name) == expected for name,expected in image_hashes.items())
    paths = [ROOT/name for name in plan['remove_root_py']]
    paths += [ROOT/'results'/name for name in plan['remove_results_dirs']]
    paths += [ROOT/name for name in plan['remove_other_obsolete_paths']]
    # These are transport archives, not the canonical imported results or
    # cross-host replay. The full original source archive remains preserved.
    transport = [p for p in (HERE/'runtime_validation').glob('*incoming') if p.is_dir()]
    duplicate = HERE/'runtime_validation'/'remote_main512_output.tar.gz'
    if duplicate.exists():
        transport.append(duplicate)
    paths += transport
    # Remove only regenerable project caches, never user environments or
    # tool configuration. Scientific results and source snapshots are kept.
    paths += [ROOT/'__pycache__', ROOT/'.pytest_cache']
    for base in [HERE, ROOT/'results'/'baseline_ab128_32_ratio12_20260912', ROOT/'docs']:
        paths += list(base.rglob('__pycache__'))
    existing = []
    for path in sorted(set(p for p in paths if p.exists()), key=lambda p:len(p.parts)):
        if not any(path.is_relative_to(parent) for parent in existing):
            existing.append(path)
    for path in existing:
        assert path.is_relative_to(ROOT) and not path.is_symlink()
    inventory = [size_info(p) for p in existing]
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'executed' if args.execute else 'dry_run',
              'plan_sha256':sha(HERE/'cleanup_plan.json'),
              'executor_sha256':sha(Path(__file__)),
              'pending_commands':pending, 'removed_or_proposed':inventory,
              'files':sum(r['files'] for r in inventory), 'bytes':sum(r['bytes'] for r in inventory),
              'kept_root_source_and_test_hashes_equal':True,
              'retained_test_result':{'passed':159,'source_record_sha256':sha(HERE/'retained_tests_check.json')},
              'saturday_preexisting_image_count':len(image_hashes),
              'saturday_preexisting_image_hashes_equal':True,
              'remote_acceptance_hashes':acceptances,
              'final_research_audit_sha256':sha(final_path) if args.execute else None,
              'historical_recovery':'Previously tracked older material is in 38edfa31; all original root Python/data sources also have the retained source archive. No claim is made that every deleted untracked artifact exists in Git history.'}
    if args.execute:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        assert set(p.name for p in (ROOT/'results').iterdir() if p.is_dir()) == set(plan['keep_results'])
        assert all(sha(ROOT/name) == expected for name,expected in tests['source_sha256'].items())
        assert all(sha(ROOT/name) == expected for name,expected in image_hashes.items())
        (HERE/'cleanup_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        lines = ['# 项目清理记录', '',
                 '按已授权范围执行：results仅保留周六研究和本次研究。核心模拟器、策略、data及原有周六图片字节未变。', '',
                 f'清理 {record["files"]:,} 个文件，共 {record["bytes"]/1024**3:.3f} GiB；包括旧研究目录、54个旧根目录脚本、已验收的重复传输归档及可再生成的项目缓存。', '',
                 '保留全部本次科学结果（含高利用率、覆盖不通过的种子），原始输入、源快照、失败运行证据和独立审计。35项运行文件及16个测试模块与159项测试通过时的SHA逐项一致。', '',
                 f'清理前已存在的周六 PNG/PDF/SVG 共 {len(image_hashes)} 个，前后SHA全部相同。', '',
                 '较早已入库资料可从38edfa31恢复；本次不承诺每个旧的未入库中间产物都在Git历史中。逐路径数量、大小与校验记录见 [JSON](cleanup_execution.json)。']
        (HERE/'cleanup_execution.md').write_text('\n'.join(lines)+'\n')
    else:
        (HERE/'cleanup_dry_run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'files':record['files'],'GiB':record['bytes']/1024**3,'pending':len(pending)}))


if __name__ == '__main__':
    main()
