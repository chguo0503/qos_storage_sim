// Three independent continuous-service SSDs, exact manifest volumes, aggregate NPU link.
// This remains a fluid proxy: no individual 176-KiB nonpreemptive commands.
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
using namespace std;
constexpr int N=32,S=3,L=8;
constexpr double EPS=1e-10,INF=1e100;
struct Request {int role;double c;array<double,S> v;};
struct Card {vector<Request> q;int r=0,l=0,ready=-1,read=-1;array<double,S> rem={0,0,0};double buf=0,ce=INF;bool done=false;};
struct Window {double lo,hi,compute[N][2]={{0}},under=0,maxd[S]={0,0,0};int maxa=0,mina=32,minactive=32;};
int main(){
 double horizon,scale[2];int link;if(!(cin>>horizon>>link>>scale[0]>>scale[1]))return 1;array<Card,N>a;
 for(auto &x:a){int n;cin>>n;x.q.resize(n);for(auto &q:x.q){cin>>q.role>>q.c;q.c*=scale[q.role];for(double &v:q.v)cin>>v;}x.read=0;x.rem=x.q[0].v;}
 vector<Window>w={{2000,4000},{2000,6000},{4000,8000}};
 if(horizon>8000)w.push_back({8000,min(16000.,horizon)});
 if(horizon>16000)w.push_back({16000,min(32000.,horizon)});
 if(horizon>32000)w.push_back({32000,horizon});
 bool traceall=getenv("OD_PROXY_TRACE_ALL")!=nullptr;
 ofstream trace;if(const char*p=getenv("OD_PROXY_TRACE")){trace.open(p);trace<<"time_ms,event,npu,request,layer,role\n"<<setprecision(14);}
 double t=0,cap=.04,base=.00125,linkrate=.05,overms=0,physical[S]={0,0,0},maxd[S]={0,0,0},maxbuf=0;long events=0;int maxa=0,minalive=32;
 auto start=[&](int i){auto &x=a[i];if(x.done||x.ce<INF/2||x.ready!=x.r*L+x.l)return;
  x.ready=-1;x.ce=t+x.q[x.r].c;
  if(trace&&i%8==0&&(x.l==0||traceall))trace<<t<<",compute_start,"<<i<<","<<x.r<<","<<x.l<<","<<x.q[x.r].role<<"\n";
  int nr=x.r,nl=x.l+1;if(nl==L){nr++;nl=0;}
  if(nr<(int)x.q.size()){if(x.read!=-1){cerr<<"duplicate read";exit(2);}x.read=nr*L+nl;x.rem=x.q[nr].v;x.buf=0;}
 };
 while(t<horizon-EPS){
  double rate[N][S]={{0}},inflow[N]={0},drain[N]={0},demand[S]={0,0,0},dt=horizon-t;int ng[S][8]={{0}},active[S]={0},groups[S]={0},ka=0,alive=0;
  for(int i=0;i<N;i++){auto &x=a[i];if(!x.done){alive++;ka+=x.q[x.r].role==0;for(int s=0;s<S;s++)demand[s]+=x.q[x.r].v[s]/x.q[x.r].c*1000;}
   for(int s=0;s<S;s++)if(x.read>=0&&x.rem[s]>EPS){ng[s][i%8]++;active[s]++;}
   if(x.ce<INF/2)dt=min(dt,max(0.,x.ce-t));
  }
  for(int s=0;s<S;s++)for(int g:ng[s])groups[s]+=g>0;
  maxa=max(maxa,ka);minalive=min(minalive,alive);
  for(int i=0;i<N;i++){auto &x=a[i];if(x.read<0)continue;
   for(int s=0;s<S;s++)if(x.rem[s]>EPS){rate[i][s]=base+(cap-base*active[s])/groups[s]/ng[s][i%8];dt=min(dt,x.rem[s]/rate[i][s]);inflow[i]+=rate[i][s];}
   drain[i]=link?(x.buf>EPS?linkrate:min(linkrate,inflow[i])):inflow[i];
   if(x.buf>EPS&&drain[i]>inflow[i]+EPS)dt=min(dt,x.buf/(drain[i]-inflow[i]));
  }
  if(dt<1e-12){cerr<<"zero event "<<t;return 3;}
  bool under=true;for(int s=0;s<S;s++){maxd[s]=max(maxd[s],demand[s]);under&=demand[s]<40-1e-9;}if(!under)overms+=dt;
  for(auto &z:w){double d=max(0.,min(t+dt,z.hi)-max(t,z.lo));if(!d)continue;z.maxa=max(z.maxa,ka);z.mina=min(z.mina,ka);z.minactive=min(z.minactive,alive);if(under)z.under+=d;
   for(int s=0;s<S;s++)z.maxd[s]=max(z.maxd[s],demand[s]);
   for(int i=0;i<N;i++)if(a[i].ce<INF/2)z.compute[i][a[i].q[a[i].r].role]+=d;
  }
  for(int i=0;i<N;i++){auto &x=a[i];if(x.read>=0){for(int s=0;s<S;s++){x.rem[s]=max(0.,x.rem[s]-rate[i][s]*dt);physical[s]+=rate[i][s]*dt;}
    x.buf=max(0.,x.buf+(inflow[i]-drain[i])*dt);maxbuf=max(maxbuf,x.buf);}}
  t+=dt;events++;
  for(int i=0;i<N;i++){auto &x=a[i];if(x.read>=0&&x.rem[0]<=EPS&&x.rem[1]<=EPS&&x.rem[2]<=EPS&&x.buf<=EPS){if(trace&&traceall&&i%8==0)trace<<t<<",io_ready,"<<i<<","<<x.read/L<<","<<x.read%L<<","<<x.q[x.read/L].role<<"\n";x.ready=x.read;x.read=-1;x.rem={0,0,0};x.buf=0;}}
  for(int i=0;i<N;i++){auto &x=a[i];if(x.ce<=t+EPS){x.ce=INF;x.l++;if(x.l==L){x.r++;x.l=0;if(x.r==(int)x.q.size())x.done=true;else if(trace&&i%8==0)trace<<t<<",admission,"<<i<<","<<x.r<<",0,"<<x.q[x.r].role<<"\n";}}}
  for(int i=0;i<N;i++)start(i);
 }
 cout<<setprecision(12)<<"{\"events\":"<<events<<",\"max_A_all\":"<<maxa<<",\"min_active_all\":"<<minalive<<",\"over_ms_all\":"<<overms<<",\"max_demand_by_ssu\":["<<maxd[0]<<","<<maxd[1]<<","<<maxd[2]<<"],\"ssd_GiB_by_ssu\":["<<physical[0]<<","<<physical[1]<<","<<physical[2]<<"],\"max_link_buffer_GiB\":"<<maxbuf<<",\"windows\":[";
 for(int k=0;k<(int)w.size();k++){auto &z=w[k];double sum=0,mina=INF,minb=INF;int mixed=0;for(int i=0;i<N;i++){sum+=z.compute[i][0]+z.compute[i][1];mina=min(mina,z.compute[i][0]);minb=min(minb,z.compute[i][1]);mixed+=z.compute[i][0]>EPS&&z.compute[i][1]>EPS;}
  if(k)cout<<",";cout<<"{\"lo\":"<<z.lo<<",\"hi\":"<<z.hi<<",\"U\":"<<sum/(N*(z.hi-z.lo))<<",\"mixed_cards\":"<<mixed<<",\"min_A_compute_ms\":"<<mina<<",\"min_B_compute_ms\":"<<minb<<",\"max_A\":"<<z.maxa<<",\"min_A\":"<<z.mina<<",\"min_active\":"<<z.minactive<<",\"under_fraction\":"<<z.under/(z.hi-z.lo)<<",\"max_demand_by_ssu\":["<<z.maxd[0]<<","<<z.maxd[1]<<","<<z.maxd[2]<<"],\"per_npu_compute_ms\":[";
  for(int i=0;i<N;i++){if(i)cout<<",";cout<<"["<<z.compute[i][0]<<","<<z.compute[i][1]<<"]";}cout<<"]}";
 }
 cout<<"]}\n";
}
