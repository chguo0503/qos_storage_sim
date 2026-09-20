// A search proxy, NOT the discrete-command simulator. One balanced reference SSD.
// OD CIR plus native two-level residual sharing, one I/O ahead, one request/NPU.
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace std;
constexpr int N=32,L=8;
constexpr double EPS=1e-10,INF=1e100;
struct Card { vector<int> q; int r=0,l=0,ready=-1,read=-1; double rem=0,buf=0,ce=INF; bool done=false; };
struct Window {double lo,hi,compute[32][2]={{0}},active=0,under=0;int maxa=0,mina=32,minactive=32;};
int main(){
 double C[2],V[2],horizon;int link;
 if(!(cin>>C[0]>>V[0]>>C[1]>>V[1]>>horizon>>link)) return 1;
 array<Card,N> a;
 for(auto &x:a){int count;cin>>count;x.q.resize(count);for(auto &v:x.q)cin>>v;x.read=0;x.rem=V[x.q[0]];}
 ofstream trace;
 if(const char* path=getenv("OD_PROXY_TRACE")){trace.open(path);trace<<"time_ms,event,npu,request,layer,role\n"<<setprecision(14);for(int i=0;i<N;i+=8)trace<<"0,admission,"<<i<<",0,0,"<<a[i].q[0]<<"\n";}
 vector<Window> w={{2000,4000},{2000,6000},{4000,8000}};
 if(horizon>8000)w.push_back({8000,min(16000.,horizon)});
 if(horizon>16000)w.push_back({16000,min(32000.,horizon)});
 if(horizon>32000)w.push_back({32000,horizon});
 double t=0,capacity=40.0/1000,base=1.25/1000,linkrate=50.0/3/1000;
 long events=0;int maxaall=0,minactiveall=32;double overms=0,physical=0,maxqueue=0;
 auto start=[&](int i){auto &x=a[i];if(x.done||x.ce<INF/2||x.ready!=x.r*L+x.l)return;
   x.ready=-1; x.ce=t+C[x.q[x.r]];
   if(trace&&i%8==0&&x.l==0)trace<<t<<",compute_start,"<<i<<","<<x.r<<","<<x.l<<","<<x.q[x.r]<<"\n";
   int nr=x.r,nl=x.l+1;if(nl==L){nr++;nl=0;}
   if(nr<(int)x.q.size()){if(x.read!=-1){cerr<<"duplicate read";exit(2);}x.read=nr*L+nl;x.rem=V[x.q[nr]];x.buf=0;}
 };
 while(t<horizon-EPS){
   double rate[N]={0},drain[N]={0},dt=horizon-t;int ng[8]={0},active=0,groups=0,ka=0,alive=0;
   for(int i=0;i<N;i++){auto &x=a[i];if(!x.done){alive++;ka+=x.q[x.r]==0;}if(x.read>=0&&x.rem>EPS){ng[i%8]++;active++;}if(x.ce<INF/2)dt=min(dt,max(0.,x.ce-t));}
   for(int g:ng)groups+=g>0;
   maxaall=max(maxaall,ka);minactiveall=min(minactiveall,alive);
   for(int i=0;i<N;i++){
     auto &x=a[i];
     if(x.read<0)continue;
     if(x.rem>EPS){rate[i]=base+(capacity-base*active)/groups/ng[i%8];dt=min(dt,x.rem/rate[i]);}
     drain[i]=link?(x.buf>EPS?linkrate:min(linkrate,rate[i])):rate[i];
     if(x.buf>EPS&&drain[i]>rate[i]+EPS)dt=min(dt,x.buf/(drain[i]-rate[i]));
   }
   if(dt<1e-12){cerr<<"zero event "<<t<<"\n";return 3;}
   double demand=ka*V[0]/C[0]*1000+(alive-ka)*V[1]/C[1]*1000;
   if(demand>40+1e-9)overms+=dt;
   for(auto &z:w){double d=max(0.,min(t+dt,z.hi)-max(t,z.lo));if(!d)continue;
     z.maxa=max(z.maxa,ka);z.mina=min(z.mina,ka);z.minactive=min(z.minactive,alive);if(demand<40-1e-9)z.under+=d;z.active+=d*alive;
     for(int i=0;i<N;i++)if(a[i].ce<INF/2)z.compute[i][a[i].q[a[i].r]]+=d;
   }
   for(int i=0;i<N;i++){auto &x=a[i];if(x.read>=0){x.rem=max(0.,x.rem-rate[i]*dt);x.buf=max(0.,x.buf+(rate[i]-drain[i])*dt);physical+=rate[i]*dt;maxqueue=max(maxqueue,x.buf);}}
   t+=dt;events++;
   for(auto &x:a)if(x.read>=0&&x.rem<=EPS&&x.buf<=EPS){x.ready=x.read;x.read=-1;x.rem=x.buf=0;}
   for(int i=0;i<N;i++){auto &x=a[i];if(x.ce<=t+EPS){x.ce=INF;x.l++;if(x.l==L){x.r++;x.l=0;if(x.r==(int)x.q.size())x.done=true;else if(trace&&i%8==0)trace<<t<<",admission,"<<i<<","<<x.r<<",0,"<<x.q[x.r]<<"\n";}}}
   for(int i=0;i<N;i++)start(i);
 }
 cout<<setprecision(12)<<"{\"events\":"<<events<<",\"max_A_all\":"<<maxaall<<",\"min_active_all\":"<<minactiveall<<",\"over_ms_all\":"<<overms<<",\"ssd_GiB\":"<<physical<<",\"max_link_buffer_GiB\":"<<maxqueue<<",\"windows\":[";
 for(int k=0;k<(int)w.size();k++){auto &z=w[k];double sum=0,mina=INF,minb=INF;int mixed=0;for(int i=0;i<N;i++){sum+=z.compute[i][0]+z.compute[i][1];mina=min(mina,z.compute[i][0]);minb=min(minb,z.compute[i][1]);mixed+=z.compute[i][0]>EPS&&z.compute[i][1]>EPS;}
   if(k)cout<<",";cout<<"{\"lo\":"<<z.lo<<",\"hi\":"<<z.hi<<",\"U\":"<<sum/(N*(z.hi-z.lo))<<",\"mixed_cards\":"<<mixed<<",\"min_A_compute_ms\":"<<mina<<",\"min_B_compute_ms\":"<<minb<<",\"max_A\":"<<z.maxa<<",\"min_A\":"<<z.mina<<",\"min_active\":"<<z.minactive<<",\"under_fraction\":"<<z.under/(z.hi-z.lo)<<",\"per_npu_compute_ms\":[";
   for(int i=0;i<N;i++){if(i)cout<<",";cout<<"["<<z.compute[i][0]<<","<<z.compute[i][1]<<"]";}cout<<"]}";
 }
 cout<<"]}\n";
}
