#include "../src/gumbel.cpp"
#include <cassert>
#include <iostream>
#include <functional>
int main(){
 // Exhaust all player-transition patterns up to four edges and terminal values.
 for(int depth=1;depth<=4;++depth)for(int mask=0;mask<(1<<(depth+1));++mask)for(double value:{-1.,-.25,0.,.75,1.}){
  gumbel::Tree tree(0);std::vector<gumbel::Node> nodes(depth+1);gumbel::Path path;
  for(int i=0;i<=depth;++i)nodes[i].player=(mask>>i)&1;
  path.leaf=&nodes.back();for(int i=0;i<depth;++i){nodes[i].edges.emplace_back();nodes[i].edges[0].pending=1;path.edges.emplace_back(&nodes[i],0);}
  tree.backup(path,value);
  for(int i=0;i<depth;++i){auto& e=nodes[i].edges[0];assert(e.visits==1 && e.pending==0);assert(e.sum==(nodes[i].player==nodes.back().player?value:-value));}
 }
 // Independently enumerate partial binary proof trees. A numerical +-1 is deliberately used for every
 // UNKNOWN leaf; it must never supply a proof. Include consecutive placements by the same player and
 // late, outstanding backups after an ancestor has already become exact.
 for(int depth=1;depth<=3;++depth)for(int players=0;players<(1<<(depth+1));++players){
  int leaves=1<<depth,states=1;for(int i=0;i<leaves;++i)states*=3;
  for(int assignment=0;assignment<states;++assignment){
   gumbel::Tree t(0);std::vector<gumbel::Node*> level{t.root.get()};std::vector<std::vector<std::pair<gumbel::Node*,int>>> paths(1);
   for(int d=0;d<depth;++d){std::vector<gumbel::Node*> next;std::vector<std::vector<std::pair<gumbel::Node*,int>>> next_paths;
    for(int i=0;i<int(level.size());++i){auto* node=level[i];node->player=(players>>d)&1;node->expanded=true;
     for(int j=0;j<2;++j){gumbel::Edge e;e.prior=.5;e.child=std::make_unique<gumbel::Node>();next.push_back(e.child.get());node->edges.push_back(std::move(e));auto p=paths[i];p.emplace_back(node,j);next_paths.push_back(std::move(p));}
    }level=std::move(next);paths=std::move(next_paths);
   }
   std::vector<int> observed(leaves,-1);int encoded=assignment;
   std::function<int(int,int)> expected=[&](int d,int offset){
    if(d==depth)return observed[offset];int player=(players>>d)&1;
    int a=expected(d+1,offset),b=expected(d+1,offset+(1<<(depth-d-1)));
    return a==player || b==player?player:a==1-player && b==1-player?1-player:-1;
   };
   for(int i=0;i<leaves;++i){int verdict=encoded%3-1;encoded/=3;observed[i]=verdict;level[i]->player=(players>>depth)&1;level[i]->exact_winner=verdict;
    gumbel::Path p;p.leaf=level[i];p.edges=paths[i];for(auto [node,index]:p.edges)++node->edges[index].pending;
    t.backup(p,i%2?1:-1);assert(t.root->exact_winner==expected(0,0));
    for(auto [node,index]:p.edges){auto& e=node->edges[index];assert(e.pending==0);assert(e.exact_winner==e.child->exact_winner);if(e.exact_winner>=0)assert(e.sum==(e.exact_winner==node->player?e.visits:-e.visits));}
   }
  }
 }
 // Neither an empty nor an unexpanded action list can certify a loss.
 {gumbel::Tree t(0);gumbel::Node node;node.player=0;t.settle(node);assert(node.exact_winner==-1);node.edges.emplace_back();node.edges[0].exact_winner=1;t.settle(node);assert(node.exact_winner==-1);}
 gumbel::Tree tree(1);tree.advance({0,0});tree.begin(16,4);
 assert(tree.sequence==std::vector<int>({0,0,0,0,1,1,1,1,2,2,3,3,4,4,5,5}));
 gumbel::Node n;n.value=.2;
 for(int i=0;i<3;++i){gumbel::Edge e;e.prior=1./3;e.visits=i==0?2:0;e.sum=i==0?1.:0;n.edges.push_back(std::move(e));}
 auto q=tree.transformed(n);assert(std::abs(q[0]-5.2)<1e-10 && q[1]==0 && q[2]==0);
 // A proven loss (ineligible, Q -1 with more visits) sets neither the range, the mixed value nor the visit scale.
 {gumbel::Node m;m.value=.2;m.player=0;
  for(int i=0;i<3;++i){gumbel::Edge e;e.prior=1./3;e.visits=i<2?2:9;e.sum=i==0?1.:-1.;m.edges.push_back(std::move(e));}
  m.edges[2].exact_winner=1;m.edges[2].eligible=false;
  auto r=tree.transformed(m);assert(std::abs(r[0]-5.2)<1e-10 && std::abs(r[1])<1e-12);}
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
 std::cout<<"Exhaustive backup signs and partial proof trees, sequential-halving schedule and mixed-Q transform passed\n";
}
