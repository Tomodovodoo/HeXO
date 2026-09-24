#include "../src/gumbel.cpp"
#include <cassert>
#include <iostream>
int main(){
 // Exhaust all player-transition patterns up to four edges and terminal values.
 for(int depth=1;depth<=4;++depth)for(int mask=0;mask<(1<<(depth+1));++mask)for(double value:{-1.,-.25,0.,.75,1.}){
  gumbel::Tree tree(0);std::vector<gumbel::Node> nodes(depth+1);gumbel::Path path;
  for(int i=0;i<=depth;++i)nodes[i].player=(mask>>i)&1;
  path.leaf=&nodes.back();for(int i=0;i<depth;++i){nodes[i].edges.emplace_back();nodes[i].edges[0].pending=1;path.edges.emplace_back(&nodes[i],0);}
  tree.backup(path,value);
  for(int i=0;i<depth;++i){auto& e=nodes[i].edges[0];assert(e.visits==1 && e.pending==0);assert(e.sum==(nodes[i].player==nodes.back().player?value:-value));}
 }
 gumbel::Tree tree(1);tree.advance({0,0});tree.begin(16,4);
 assert(tree.sequence==std::vector<int>({0,0,0,0,1,1,1,1,2,2,3,3,4,4,5,5}));
 gumbel::Node n;n.value=.2;
 for(int i=0;i<3;++i){gumbel::Edge e;e.prior=1./3;e.visits=i==0?2:0;e.sum=i==0?1.:0;n.edges.push_back(std::move(e));}
 auto q=tree.transformed(n);assert(std::abs(q[0]-5.2)<1e-10 && q[1]==0 && q[2]==0);
 // Final selection excludes eliminated actions despite a larger stale score.
 tree.root->expanded=true;tree.root->value=0;
 for(int i=0;i<2;++i){gumbel::Edge e;e.action={i,1};e.prior=.5;e.logit=i?0:100;e.epoch=i?4:1;tree.root->edges.push_back(std::move(e));}
 int64_t actions[4];int visits[2];double values[2],scores[2];
 hxg_stats(&tree,actions,visits,values,scores);
 assert(!std::isfinite(scores[0]) && std::isfinite(scores[1]));
 // Reused estimates must not change the initial policy Gumbel sample.
 tree.root->edges[0].logit=1;tree.root->edges[0].gumbel=0;tree.root->edges[0].visits=10000;tree.root->edges[0].sum=-10000;
 tree.root->edges[1].logit=0;tree.root->edges[1].gumbel=0;tree.root->edges[1].visits=10000;tree.root->edges[1].sum=10000;
 for(auto& e:tree.root->edges)e.epoch=0;
 int request=tree.request();assert(request>0 && tree.requests.at(request).edges.front().second==0);tree.cancel();
 std::cout<<"Exhaustive backup sign patterns, sequential-halving schedule and mixed-Q transform passed\n";
}
