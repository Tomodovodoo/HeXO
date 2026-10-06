#include "../src/gumbel.cpp"
#include <cassert>
#include <iostream>
#include <functional>
int main(){
 // Continuation identities include coloured stones, placement phase and the
 // network's last-turn inputs. Incremental keys must identify the actual board.
 {Board board;std::vector<Cell> history;std::mt19937_64 rng(31);
  for(int stones=0;stones<96 && board.winner<0;++stones){
   auto legal=board.legal_moves();auto parent=gumbel::keys(board).first;
   for(size_t i=0;i<std::min(size_t(17),legal.size());++i){auto action=legal[i*legal.size()/std::min(size_t(17),legal.size())];
    Board child=board;child.make(action);
    assert(gumbel::child_keys(parent,history,action)==gumbel::keys(child));
    assert(gumbel::predecessor(gumbel::keys(child).first,history.size()+1,action)==parent);
   }
   auto action=legal[rng()%legal.size()];board.make(action);history.push_back(action);
  }
 }
 // One pass over a history gives every prefix the keys of its replayed board.
 {Board board;std::vector<Cell> history;std::mt19937_64 rng(37);
  for(int stones=0;stones<80 && board.winner<0;++stones){auto legal=board.legal_moves();auto action=legal[rng()%legal.size()];board.make(action);history.push_back(action);}
  auto all=gumbel::prefix_keys(history);assert(all.size()==history.size()+1);Board replay;
  for(size_t k=0;k<=history.size();++k){
   assert(all[k]==gumbel::keys(replay) && all[k]==gumbel::keys(std::vector<Cell>(history.begin(),history.begin()+k)));
   if(k<history.size())replay.make(history[k]);
  }
 }
 // Stored continuations reconnect through both legal orders of a turn, but
 // rule-position identity alone cannot alias different neural turn inputs.
 {gumbel::Tree t(19);assert(hxg_share(&t,256));
  auto expand=[&](const std::vector<Cell>& h){t.root_at(h);t.begin(1,1);int id=t.request();assert(id>0);
   auto legal=t.requests.at(id).legal;std::vector<int64_t> actions;std::vector<double> logits(legal.size()),values(legal.size());
   for(size_t i=0;i<legal.size();++i){actions.push_back(legal[i].q);actions.push_back(legal[i].r);logits[i]=-double(i)/17;values[i]=i%2?.7:-.4;}
   t.fulfill(id,actions.data(),logits.data(),values.data(),int(legal.size()));
   double mass=0,value=0;for(size_t i=0;i<legal.size();++i){double w=std::exp(logits[i]);mass+=w;value+=w*values[i];}
   assert(std::abs(t.root->value-value/mass)<1e-12);assert(t.root->edges.size()==legal.size());
   for(size_t i=0;i<legal.size();++i)assert(t.root->edges[i].action==legal[i] && std::abs(t.root->prior(t.root->edges[i])-std::exp(logits[i])/mass)<1e-12);
   return t.root;
  };
  auto edge=[](auto& n,Cell action)->gumbel::Edge&{auto e=std::find_if(n->edges.begin(),n->edges.end(),[&](auto& e){return e.action==action;});assert(e!=n->edges.end());return *e;};
  auto completed=expand({{0,0},{1,0},{2,0}});
  auto first=expand({{0,0},{1,0}});assert(edge(first,{2,0}).read().child==completed);
  auto reversed=expand({{0,0},{2,0}});assert(edge(reversed,{1,0}).read().child==completed);
  t.root_at({{0,0},{1,0},{2,0}});t.mark(t.root->edges.front().action,t.root->player,5);
  assert(edge(first,{2,0}).read().exact_winner==completed->player && edge(reversed,{1,0}).read().exact_winner==completed->player);
  std::vector<Cell> a{{0,0},{1,0},{2,0},{0,1},{0,2},{1,1},{2,1},{1,2},{2,2}};
  std::vector<Cell> b{{0,0},{1,0},{2,0},{1,2},{2,2},{1,1},{2,1},{0,1},{0,2}};
  auto middle=a;middle.push_back({3,0});auto child=expand(middle);
  auto wrong=expand(b);assert(wrong->position==gumbel::keys(a).first && !edge(wrong,{3,0}).read().child);
  auto right=expand(a);assert(edge(right,{3,0}).read().child==child);
  middle.push_back({4,0});auto whole=expand(middle);
  auto other=b;other.push_back({3,0});auto peer=expand(other);
  assert(peer->position==child->position && peer->context!=child->context);
  assert(edge(peer,{4,0}).read().child==whole && edge(child,{4,0}).read().child==whole);
 }
 // Round sampling remains explicit while useful work crosses visit layers.
 for(bool shared:{false,true}){gumbel::Tree t(23);if(shared)assert(hxg_share(&t,256));t.advance({0,0});t.begin(128,8);
  auto install=[&](int id){auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
   std::vector<double> zeros(legal.size());t.fulfill(id,cells.data(),zeros.data(),zeros.data(),int(legal.size()));};
  install(t.request());assert(hxg_round_barrier(&t,1));std::vector<int> first;for(int i=0;i<8;++i){int id=t.request();assert(id>0);first.push_back(id);}
  assert(t.request()==0 && t.round.active==0 && t.started==8 && t.completed==0);
  install(first[0]);int next=t.request();assert(next>0 && t.round.active==0 && t.started==9 && t.completed==1);
  // An in-flight refutation may replace its slot, but it never transfers credit.
  int old=t.requests.at(first[1]).edges.front().second;uint64_t credits=t.root_edges[old].credits;
  t.mark(t.root->edges[old].action,1-t.root->player,4);
  int replacement=t.request();assert(replacement>0 && t.round.active==0);
  assert(std::find(t.round.members.begin(),t.round.members.end(),old)==t.round.members.end());
  assert(t.root_edges[old].credits==credits);
  install(first[1]);assert(t.root_edges[old].credits==credits+1);
  assert(!t.root->edges[old].read().child->expanded);
  // Cancelling old and replacement reservations releases exactly their slots.
  t.cancel();assert(t.started==t.completed && t.requests.empty());
  for(int c:t.round.counts)assert(c>=0);
  int loops=0;while(!t.done()){
   assert(++loops<256);std::vector<int> pending;
   for(int i=0;i<128;++i){int before=t.round.active;size_t outstanding=t.requests.size();int id=t.request();if(id>0)pending.push_back(id);else if(id==0)break;
    if(t.round.active!=before)assert(outstanding==0);}
   assert(!pending.empty() || t.done());for(auto i=pending.rbegin();i!=pending.rend();++i)install(*i);
   for(size_t i=0;i<t.round.counts.size();++i)assert(t.round.counts[i]>=0 && t.round.counts[i]<=t.round.limits[i]);
  }
  uint64_t completed=0;for(auto& e:t.root_edges)completed+=e.credits;assert(completed==128 && t.completed==128 && t.started==128);
  assert(t.requests.empty());for(auto& e:t.root->edges)assert(e.read().pending==0);
  std::vector<double> scores(t.root->edges.size()),values(scores.size());std::vector<int> visits(scores.size());std::vector<int64_t> actions(2*scores.size());
  hxg_stats(&t,actions.data(),visits.data(),values.data(),scores.data());
  for(size_t i=0;i<scores.size();++i)if(std::isfinite(scores[i]))assert(t.root->edges[i].read().eligible && std::find(t.round.members.begin(),t.round.members.end(),int(i))!=t.round.members.end());
 }
 // Changing sampling contracts invalidates old comparison sessions, not graph evidence.
 {gumbel::Tree t(23);assert(hxg_share(&t,96));t.root_at({{0,0}});t.begin(16,4);
  int id=t.request();auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}std::vector<double> zeros(legal.size());
  t.fulfill(id,cells.data(),zeros.data(),zeros.data(),int(legal.size()));int pending=t.request();assert(pending>0);t.cancel();auto original=t.root;
  t.root_at({{0,0},{2,0}});assert(hxg_round_barrier(&t,1));t.root_at({{0,0}});
  assert(t.root==original && t.root->expanded && t.budget==0 && t.started==0);
  t.begin(16,4);assert(t.request()>0);t.cancel();
 }
 // Existing own-turn completions settle before any neural request. Both stones
 // stay legal, the full action list remains present, and no reservation survives.
 for(bool shared:{false,true}){gumbel::Tree t(7);if(shared)assert(hxg_share(&t,16));t.tactics=true;
  std::vector<Cell> h{{0,0},{0,3},{1,3},{1,0},{2,0},{2,3},{3,3},{3,0},{7,4},{4,3},{5,4}};
  for(auto c:h)t.advance(c);t.begin(32,8);assert(t.request()==-1 && t.done());
  assert(t.requests.empty() && !t.root->pending && t.root->exact_winner==0 && t.root->distance==2);
  assert(t.completed==0 && t.issued==0 && t.root->edges.size()==t.board.legal_moves().size());
  for(int stone=0;stone<2;++stone){
   auto it=std::find_if(t.root->edges.begin(),t.root->edges.end(),[](auto& e){return e.read().eligible;});
   assert(it!=t.root->edges.end() && it->read().exact_winner==0 && t.board.legal(it->action));
   t.advance(it->action);t.begin(32,8);assert(t.request()<=0 && t.requests.empty());
  }
  assert(t.board.winner==0);
 }
 // An immediate interior win completes one selected simulation and comparison
 // credit. It refutes that move, not the unresolved parent, after the mover flips.
 {gumbel::Tree t(7);assert(hxg_share(&t,16));
  std::vector<Cell> h{{0,0},{0,3},{1,3},{1,0},{2,0},{2,3},{3,3},{3,0},{7,4},{4,3}};
  for(auto c:h)t.advance(c);t.begin(1,1);int id=t.request();assert(id>0);
  auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;std::vector<double> logits,q(legal.size());
  for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);logits.push_back(c==Cell{5,4}?100:-100);}
  t.fulfill(id,cells.data(),logits.data(),q.data(),int(legal.size()));t.tactics=true;
  assert(t.request()==-1 && t.done() && t.completed==1 && t.issued==1 && t.requests.empty());
  auto it=std::find_if(t.root->edges.begin(),t.root->edges.end(),[](auto& e){return e.action==Cell{5,4};});
  assert(it!=t.root->edges.end() && it->read().visits==1 && it->read().pending==0);
  assert(it->read().exact_winner==0 && it->read().distance==3 && it->read().child->distance==2);
  assert(t.root_edges[it-t.root->edges.begin()].credits==1 && t.root->exact_winner<0);
 }
 // A shared loss proof materializes every legal edge in dormant peer contexts.
 // That growth must be trimmed at delivery, without losing the exact outcome.
 for(bool leaf_proof:{false,true}){gumbel::Tree t(7);assert(hxg_share(&t,1));assert(hxg_archive(&t,65536));
  std::vector<Cell> original{{0,0},{1,1},{2,1},{2,0},{0,3},{0,2},{1,2},{-1,2},{3,1},{-1,0},{0,-1}};
  auto expand=[&](std::vector<Cell> h){t.root_at(h);t.begin(1,1);int id=t.request();assert(id>0);
   auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
   std::vector<double> z(legal.size());t.fulfill(id,cells.data(),z.data(),z.data(),int(legal.size()));return t.root;};
  auto first=expand(original);auto second_history=original;
  std::swap(second_history[5],second_history[9]);std::swap(second_history[6],second_history[10]);
  auto second=expand(second_history);auto third_history=original;
  std::swap(third_history[1],third_history[9]);std::swap(third_history[2],third_history[10]);
  auto peer=expand(third_history);t.evict();
  int request=0;
  if(leaf_proof){third_history=original;std::swap(third_history[5],third_history[9]);
   t.root_at(third_history);t.begin(1,1);request=t.request();assert(request>0);peer=t.root;}
  assert(first->dormant && second->dormant);
  auto& archive=*t.state->archive;auto discarded=archive.discarded;
  int winner=leaf_proof?peer->player:1-peer->player;
  if(leaf_proof){std::vector<int64_t> history;for(auto c:third_history){history.push_back(c.q);history.push_back(c.r);}
   auto legal=t.requests.at(request).legal;int64_t moves[4]{legal[0].q,legal[0].r,legal[1].q,legal[1].r};
   assert(hxg_prove(&t,request,history.data(),int(third_history.size()),peer->player,peer->remaining,moves,2,2));
  }else assert(hxg_prove_loss(&t,winner,7));
  assert(!archive.dirty && archive.total_bytes()<=archive.limit && archive.discarded>discarded);
  int64_t counts[10];assert(hxg_archive_stats(&t,counts) && counts[1]<=counts[2]);
  t.root_at(original);assert(t.root->exact_winner==winner);
 }
 // A long active history must not consume an empty dormant archive's byte
 // allowance. Consecutive cells alternate colours in pairs and cannot win.
 {gumbel::Archive archive(65536);std::shared_ptr<const gumbel::HistoryLink> history;
  for(int i=0;i<4096;++i)history=std::make_shared<gumbel::HistoryLink>(history,Cell{i,0});
  archive.set_focus(history);assert(archive.focus_stones()==4096 && archive.total_bytes()<=archive.limit);
  archive.focus.reset();while(history){auto before=history->before;history.reset();history=std::move(before);}
 }
 // Dormant descendants survive a cut ancestor. Proofs propagate through
 // retained links, and beginning a descendant comparison promotes/pins its
 // lineage so another view cannot free a raw backup pointer.
 {gumbel::Tree t(7);assert(hxg_share(&t,1));assert(hxg_archive(&t,65536));
  auto expand=[&](std::vector<Cell> h){t.root_at(h);t.begin(1,1);int id=t.request();assert(id>0);
   auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
   std::vector<double> z(legal.size());t.fulfill(id,cells.data(),z.data(),z.data(),int(legal.size()));return t.root;};
  auto a=expand({{0,0}}),b=expand({{0,0},{1,0}}),c=expand({{0,0},{1,0},{2,0}});
  t.root_at({{0,0}});assert(b->dormant && c->dormant);
  {gumbel::Tree view(11,t.state);view.shared=view.graph=true;view.root_at({{0,0},{1,0},{2,0}});view.begin(4,2);
   assert(!b->dormant && t.state->pinned(b.get()));t.evict();t.trim_archive();
   assert(t.nodes.at(b->context).lock()==b && t.state->pinned(b.get()));
  }
  t.evict();assert(b->dormant && c->dormant);
  c->exact_winner=1;c->distance=3;t.learn(*c);t.revise(*c);
  assert(b->exact_winner==1 && a->exact_winner==1);
  auto& archive=*t.state->archive;
  auto cut=archive.remove(archive.contexts.at(b->context));t.discard(cut);cut.reset();b.reset();
  assert(archive.contexts.contains(c->context));
  t.root_at({{0,0},{1,0},{2,0}});assert(t.root==c && !c->dormant && c->exact_winner==1);
  assert(c->edges.size()==t.board.legal_moves().size());
  assert(archive.total_bytes()<=archive.limit);
 }
 // Only opposite colours prevent forward reconvergence. Test both directions,
 // including a conflict after a missing focus cell, new archive insertions,
 // safe deferral for pending/pinned nodes, and independently retained facts.
 for(bool forward:{false,true})for(bool opposite:{false,true}){
  gumbel::Tree t(7);assert(hxg_share(&t,1));assert(hxg_archive(&t,65536));assert(hxg_archive_forward(&t,forward));
  auto expand=[&](std::vector<Cell> h){t.root_at(h);t.begin(1,1);int id=t.request();assert(id>0);
   auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
   std::vector<double> z(legal.size());t.fulfill(id,cells.data(),z.data(),z.data(),int(legal.size()));return t.root;};
  auto a=expand({{0,0}}),b=expand({{0,0},{1,0},{2,0}}),c=expand({{0,0},{1,0},{2,0},{3,0}});
  t.root_at({{0,0}});auto& archive=*t.state->archive;assert(b->dormant && c->dormant && archive.total_bytes()<archive.limit);
  c->exact_winner=1;c->distance=5;t.learn(*c);auto position=c->position;
  if(opposite)t.root_at({{0,0},{4,0},{5,0},{1,0}});else t.root_at({{0,0},{3,0}});
  assert(archive.contexts.contains(c->context)==!forward);
  assert(archive.contexts.contains(b->context)==!(forward && opposite));
  assert(t.outcomes.at(position).winner==1);
  if(forward && !opposite){b->pending=true; // A late waiter must not lose its node.
   t.root_at({{0,0},{4,0},{5,0},{1,0}});assert(archive.contexts.contains(b->context));
   b->pending=false;t.trim_archive();assert(!archive.contexts.contains(b->context));
  }
  assert(archive.total_bytes()<=archive.limit);
 }
 {gumbel::Tree t(7);assert(hxg_share(&t,1));assert(hxg_archive(&t,65536));assert(hxg_archive_forward(&t,1));
  auto expand=[&](std::vector<Cell> h){t.root_at(h);t.begin(1,1);int id=t.request();assert(id>0);
   auto legal=t.requests.at(id).legal;std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
   std::vector<double> z(legal.size());t.fulfill(id,cells.data(),z.data(),z.data(),int(legal.size()));return t.root;};
  expand({{0,0}});auto b=expand({{0,0},{1,0},{2,0}});t.root_at({{0,0}});assert(b->dormant);
  {gumbel::Tree view(11,t.state);view.shared=view.graph=true;view.root_at({{0,0},{1,0},{2,0}});view.begin(4,2);int request=view.request();assert(request>0);
   auto before=t.state->archive->focus;assert(!hxg_archive_forward(&view,0));assert(t.state->archive->forward);
   t.root_at({{0,0},{4,0},{5,0},{1,0}});assert(t.state->pinned(b.get()) && view.requests.contains(request));
   view.cancel();assert(view.requests.empty());assert(t.state->archive->focus!=before);
  }
  expand({{0,0},{4,0},{5,0},{1,0}});t.evict();t.trim_archive();assert(!t.state->archive->contexts.contains(b->context));
  // New conflicts arriving from the active store are discarded on insertion.
  assert(!t.store.contains(b->context) && !b->dormant);
 }
 // Sparse selection maximizes the complete legal stable-softmax score. Cold
 // mass remains available after dominant priors are refuted, including weights
 // that underflowed at expansion. Pending children still contribute mass but
 // cannot be selected; shared child evidence changes Q without a new edge visit.
 for(bool graph:{false,true})for(double gap:{-2.,-710.,-744.,-1000.})for(double floor:{0.,.1}){
  gumbel::Tree t(0);t.graph=graph;t.range_floor=floor;gumbel::Node n;n.player=0;n.value=.3;n.expanded=true;
  for(int i=0;i<97;++i){gumbel::Edge e(&n.empty);e.action={i,0};e.logit=i?gap+(i*17%23)*.01:0;e.weight=std::exp(e.logit);n.policy_mass+=e.weight;n.edges.push_back(std::move(e));}
  n.edges[1].write().visits=7;n.edges[1].write().sum=-.7;
  n.edges[2].write().exact_winner=1;n.edges[2].eligibility(false);
  if(graph){n.edges[4].write().child=std::make_shared<gumbel::Node>();n.edges[4].read().child->player=1;n.edges[4].read().child->n=11;n.edges[4].read().child->q=.6;}
  n.edges[5].write().pending=3;n.edges[5].write().child=std::make_shared<gumbel::Node>();n.edges[5].read().child->pending=true;
  n.edges[10].write().visits=100;n.edges[10].write().sum=70;n.edges[31].eligibility(false);n.edges[32].write().pending=2;
  for(int stage=0;stage<5;++stage){
   if(stage==1){n.edges[0].write().exact_winner=1;n.edges[0].eligibility(false);}
   if(stage==2){n.edges[64].write().visits=2;n.edges[64].write().sum=-1.;n.edges[65].write().pending=1;}
   if(stage==3){n.edges[5].read().child->pending=false;n.edges[5].write().pending=0;if(graph)n.edges[4].read().child->q=-.8;}
   if(stage==4){n.edges[10].write().visits=10000;n.edges[10].write().sum=-9999.;}
   auto full=t.transformed(n);auto sparse=t.selection_q(n);
   for(int i=0;i<int(full.size());++i)assert(std::abs(full[i]-sparse[i])<=1e-12*std::max(1.,std::abs(full[i])));
   double maximum=-std::numeric_limits<double>::infinity(),mass=0;int visits=0;
   for(int i=0;i<int(full.size());++i){auto& e=n.edges[i];visits+=e.read().visits+e.read().pending;if(e.read().eligible)maximum=std::max(maximum,full[i]+e.logit);}
   for(int i=0;i<int(full.size());++i)if(n.edges[i].read().eligible)mass+=std::exp(full[i]+n.edges[i].logit-maximum);
   std::vector<double> scores(full.size(),-std::numeric_limits<double>::infinity());double best=-std::numeric_limits<double>::infinity();
   for(int i=0;i<int(full.size());++i){auto& e=n.edges[i];if(!e.read().eligible || (e.read().child && e.read().child->pending))continue;scores[i]=std::exp(full[i]+e.logit-maximum)/mass-double(e.read().visits+e.read().pending)/(1+visits);best=std::max(best,scores[i]);}
   int chosen=t.select_interior(n,sparse);assert(chosen>=0 && std::isfinite(scores[chosen]) && best-scores[chosen]<=1e-12);
   assert(n.cold_best>=0 && n.edges[n.cold_best].read().empty && n.cold_mass>=1 && std::isfinite(n.cold_mass));
   assert(n.edges.size()==97);
  }
 }
 // Exhaust all player-transition patterns up to four edges and terminal values.
 for(int depth=1;depth<=4;++depth)for(int mask=0;mask<(1<<(depth+1));++mask)for(double value:{-1.,-.25,0.,.75,1.}){
  gumbel::Tree tree(0);std::vector<gumbel::Node> nodes(depth+1);gumbel::Path path;
  for(int i=0;i<=depth;++i)nodes[i].player=(mask>>i)&1;
  path.leaf=&nodes.back();for(int i=0;i<depth;++i){nodes[i].edges.emplace_back();nodes[i].edges[0].write().pending=1;path.edges.emplace_back(&nodes[i],0);}
  tree.backup(path,value);
  for(int i=0;i<depth;++i){auto& e=nodes[i].edges[0];assert(e.read().visits==1 && e.read().pending==0);assert(e.read().sum==(nodes[i].player==nodes.back().player?value:-value));}
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
     for(int j=0;j<2;++j){gumbel::Edge e;e.write().child=std::make_unique<gumbel::Node>();next.push_back(e.read().child.get());node->edges.push_back(std::move(e));auto p=paths[i];p.emplace_back(node,j);next_paths.push_back(std::move(p));}
    }level=std::move(next);paths=std::move(next_paths);
   }
   std::vector<int> observed(leaves,-1);int encoded=assignment;
   // (winner, distance): the shortest win at the winner's choice, the longest resistance at the loser's.
   std::function<std::pair<int,int>(int,int)> expected=[&](int d,int offset)->std::pair<int,int>{
    if(d==depth)return {observed[offset],observed[offset]<0?-1:offset%3};int player=(players>>d)&1;
    auto a=expected(d+1,offset),b=expected(d+1,offset+(1<<(depth-d-1)));
    if(a.first==player || b.first==player)return {player,1+std::min(a.first==player?a.second:1<<20,b.first==player?b.second:1<<20)};
    if(a.first==1-player && b.first==1-player)return {1-player,1+std::max(a.second,b.second)};
    return {-1,-1};
   };
   for(int i=0;i<leaves;++i){int verdict=encoded%3-1;encoded/=3;observed[i]=verdict;level[i]->player=(players>>depth)&1;level[i]->exact_winner=verdict;level[i]->distance=verdict<0?-1:i%3;
    gumbel::Path p;p.leaf=level[i];p.edges=paths[i];for(auto [node,index]:p.edges)++node->edges[index].write().pending;
    t.backup(p,i%2?1:-1);auto want=expected(0,0);assert(t.root->exact_winner==want.first);if(want.first>=0)assert(t.root->distance==want.second);
    for(auto [node,index]:p.edges){auto& e=node->edges[index];assert(e.read().pending==0);assert(e.read().exact_winner==e.read().child->exact_winner);if(e.read().exact_winner>=0)assert(e.read().sum==(e.read().exact_winner==node->player?e.read().visits:-e.read().visits));}
   }
  }
 }
 // Neither an empty nor an unexpanded action list can certify a loss.
 {gumbel::Tree t(0);gumbel::Node node;node.player=0;t.settle(node);assert(node.exact_winner==-1);node.edges.emplace_back();node.edges[0].write().exact_winner=1;t.settle(node);assert(node.exact_winner==-1);}
 // A lost node keeps every loss that may resist longest. With tactics a bounded loss outlasts every one-turn loss
 // (at least remaining + 5 placements); without tactics a bound may hide a faster loss, so it only drops losses
 // that are certainly shorter.
 for(bool tactics:{true,false}){gumbel::Tree t(0);t.tactics=tactics;gumbel::Node n;n.player=0;n.remaining=1;n.expanded=true;
  int d[]={3,2,20,7};bool b[]={false,false,true,true};
  for(int i=0;i<4;++i){gumbel::Edge e;e.write().exact_winner=1;e.write().distance=d[i];e.write().bound=b[i];n.edges.push_back(std::move(e));}
  t.settle(n);assert(n.exact_winner==1 && n.distance==20 && n.bound);
  assert(n.edges[0].read().eligible==!tactics && !n.edges[1].read().eligible && n.edges[2].read().eligible && n.edges[3].read().eligible);}
 // A won node's distance is exact when an exact edge attains it and no bounded win could be faster.
 for(int slow:{20,4}){gumbel::Tree t(0);t.tactics=true;gumbel::Node n;n.player=0;n.remaining=1;n.expanded=true;
  int d[]={2,slow};bool b[]={false,true};
  for(int i=0;i<2;++i){gumbel::Edge e;e.write().exact_winner=0;e.write().distance=d[i];e.write().bound=b[i];n.edges.push_back(std::move(e));}
  t.settle(n);assert(n.exact_winner==0 && n.distance==2 && !n.bound && n.edges[0].read().eligible && !n.edges[1].read().eligible);}
 {gumbel::Tree t(0);t.tactics=true;gumbel::Node n;n.player=0;n.remaining=1;n.expanded=true;
  int d[]={9,20};bool b[]={false,true};
  for(int i=0;i<2;++i){gumbel::Edge e;e.write().exact_winner=0;e.write().distance=d[i];e.write().bound=b[i];n.edges.push_back(std::move(e));}
  t.settle(n);assert(n.distance==9 && n.bound);}
 // Graph search: both orders of each turn meet in one node; another turn partition of the same stones is another
 // context but inherits the proven outcome of the position.
 {gumbel::Tree t(0);t.graph=true;
  auto play=[&](std::vector<Cell> moves){t.board=Board{};for(auto c:moves)t.board.make(c);return t.child_here();};
  auto first=play({{0,0},{1,0},{2,0},{0,1},{0,2},{3,0},{4,0}});
  auto live=play({{0,0},{1,0},{3,0},{0,1},{0,2},{2,0},{4,0}});
  first->exact_winner=1;first->distance=3;t.learn(*first);
  assert(live!=first && live->exact_winner==1 && live->distance==3);   // an existing context learns it too
  assert(play({{0,0},{2,0},{1,0},{0,2},{0,1},{4,0},{3,0}})==first);
  auto other=play({{0,0},{3,0},{4,0},{0,1},{0,2},{1,0},{2,0}});
  assert(other!=first && other->exact_winner==1 && other->distance==3);}
 // A shared outcome installs its witness on an expanded node of the position, and an exact distance refines a
 // same-distance bound on an unexpanded one.
 {gumbel::Tree t(0);t.graph=true;gumbel::Node n;n.player=0;n.expanded=true;
  for(int i=0;i<2;++i){gumbel::Edge e;e.action={i,1};n.edges.push_back(std::move(e));}
  gumbel::Outcome won{0,0,5,3,true,{{{1,1},0,5,true}}};
  assert(t.apply(won,n) && n.exact_winner==0 && n.distance==5 && !n.edges[0].read().eligible && n.edges[1].read().eligible);
  gumbel::Node u;u.player=1;u.exact_winner=0;u.distance=4;u.bound=true;
  assert(t.apply(gumbel::Outcome{1,0,4,3,false},u) && !u.bound && !t.apply(gumbel::Outcome{1,0,6,3,false},u));}
 // Advancing into another context of a position proven won with a witness keeps the verdict; the root still expands
 // on the first request, which installs the witness as its only eligible move.
 {gumbel::Tree u(0);u.graph=true;
  for(auto c:std::vector<Cell>{{0,0},{1,0},{3,0},{0,1},{0,2},{4,0}})u.advance(c);
  u.board.make({2,0});auto position=gumbel::keys(u.board).first;u.board.undo();
  u.outcomes[position]=gumbel::Outcome{0,0,3,7,true,{{{5,0},0,3,true}}};u.advance({2,0});
  assert(!u.root->expanded && u.root->exact_winner==0);
  u.begin(8,4);int id=u.request();assert(id>0 && u.requests.at(id).edges.empty());
  auto& legal=u.requests.at(id).legal;std::vector<int64_t> a;for(auto c:legal){a.push_back(c.q);a.push_back(c.r);}
  std::vector<double> z(legal.size());u.fulfill(id,a.data(),z.data(),z.data(),int(legal.size()));
  assert(u.done() && u.root->exact_winner==0);
  for(auto& e:u.root->edges)assert(e.read().eligible==(e.action==Cell{5,0}));}
 // A second equally short win proven in one expanded context reaches an expanded peer, though the outcome is unchanged.
 {gumbel::Tree t(0);t.graph=true;
  auto make=[&](std::vector<Cell> moves){t.board=Board{};for(auto c:moves)t.board.make(c);auto n=t.child_here();n->expanded=true;
   for(int i=0;i<3;++i){gumbel::Edge e;e.action={9,i};n->edges.push_back(std::move(e));}return n;};
  auto one=make({{0,0},{1,0},{2,0},{0,1},{0,2},{3,0},{4,0}}),two=make({{0,0},{1,0},{3,0},{0,1},{0,2},{2,0},{4,0}});
  for(int i:{0,1}){one->edges[i].write().exact_winner=0;one->edges[i].write().distance=2;t.settle(*one);t.learn(*one);}
  assert(two->exact_winner==0 && two->edges[0].read().eligible && two->edges[1].read().eligible && !two->edges[2].read().eligible);}
 // A two-stone certificate whose witness context already has a node gives that node the second stone.
 {gumbel::Tree t(0);t.graph=true;t.root->position=gumbel::keys(t.board).first;
  for(auto c:std::vector<Cell>{{0,0},{1,0},{2,0}})t.advance(c);
  t.board.make({3,3});auto existing=t.child_here();t.board.undo();
  t.begin(4,2);int id=t.request();auto h=t.requests.at(id).history;std::vector<int64_t> hist;for(auto c:h){hist.push_back(c.q);hist.push_back(c.r);}
  int64_t moves[]={3,3,3,4};t.prove(id,hist.data(),int(h.size()),0,2,moves,2,2);
  auto edge=std::find_if(t.root->edges.begin(),t.root->edges.end(),[](auto& e){return e.action==Cell{3,3};});
  assert(edge->read().child==existing && existing->exact_winner==0 && existing->distance==5 && existing->bound);
  // Playing the first stone keeps the verdict, and the first request expands the root with the second stone.
  t.advance({3,3});assert(t.root==existing && t.root->exact_winner==0);
  t.begin(4,2);id=t.request();auto& l=t.requests.at(id).legal;std::vector<int64_t> a;for(auto c:l){a.push_back(c.q);a.push_back(c.r);}
  std::vector<double> z(l.size());t.fulfill(id,a.data(),z.data(),z.data(),int(l.size()));
  for(auto& e:t.root->edges)assert(e.read().eligible==(e.action==Cell{3,4}));}
 // Expanding another context of a proven loss takes a live peer's exact resistances, not only the shared bound.
 {gumbel::Tree t(0);t.graph=true;
  for(auto c:std::vector<Cell>{{0,0},{1,0},{2,0},{0,1},{0,2},{3,0}})t.advance(c);
  t.board.make({4,0});auto peer=t.child_here();auto legal=t.board.legal_moves();t.board.undo();
  peer->expanded=true;peer->remaining=2;
  for(size_t i=0;i<legal.size();++i){gumbel::Edge e;e.action=legal[i];e.write().exact_winner=1;e.write().distance=i==0?9:3;peer->edges.push_back(std::move(e));}
  t.settle(*peer);t.learn(*peer);assert(peer->exact_winner==1 && peer->distance==9 && !peer->bound);
  // With the peer alive, and after it is gone (the shared outcome keeps the per-move resistances).
  for(bool alive:{true,false}){
   gumbel::Tree u(0);u.graph=true;u.outcomes=t.outcomes;if(alive)u.positions=t.positions;
   for(auto c:std::vector<Cell>{{0,0},{1,0},{3,0},{0,1},{0,2},{2,0},{4,0}})u.advance(c);
   assert(u.root->exact_winner==1 && !u.root->expanded);
   u.begin(4,2);int id=u.request();auto& l=u.requests.at(id).legal;std::vector<int64_t> a;for(auto c:l){a.push_back(c.q);a.push_back(c.r);}
   std::vector<double> z(l.size());u.fulfill(id,a.data(),z.data(),z.data(),int(l.size()));
   assert(u.root->distance==9 && !u.root->bound && u.root->edges[0].read().eligible && !u.root->edges[1].read().eligible);
  }}
 // A longer proof from a shared child never loosens a tighter proof already on an incoming edge.
 {gumbel::Tree t(0);t.graph=true;auto p=std::make_shared<gumbel::Node>(),c=std::make_shared<gumbel::Node>();
  p->player=0;c->player=1;p->expanded=c->expanded=true;
  gumbel::Edge e;e.write().child=c;e.write().exact_winner=0;e.write().distance=3;e.write().bound=true;p->edges.push_back(std::move(e));c->parents.push_back(p);
  c->exact_winner=0;c->distance=9;t.propagate(*c,nullptr);
  assert(p->edges[0].read().distance==3 && p->edges[0].read().bound);}
 // MCGS backup: a node's value is recomputed from its edges' visits and its children's current values; a playout
 // reusing a transposed child's value leaves that child unchanged.
 {gumbel::Tree t(0);t.graph=true;gumbel::Node r;r.player=0;r.expanded=true;r.value=.2;
  auto child=std::make_shared<gumbel::Node>();child->player=1;child->expanded=true;child->n=3;child->q=.5;
  for(int i=0;i<2;++i){gumbel::Edge e;e.write().child=child;r.edges.push_back(std::move(e));}
  gumbel::Path p;p.leaf=child.get();p.edges={{&r,0}};++r.edges[0].write().pending;t.backup(p,.5,false);
  assert(r.edges[0].read().visits==1 && std::abs(r.q-(.2-.5)/2)<1e-12 && child->n==3 && r.n==1);
  assert(std::abs(t.value(r,r.edges[1])+.5)<1e-12);}
 // A backup through one parent of a shared child also brings the child's other parents up to date, verdicts included.
 {gumbel::Tree t(0);t.graph=true;
  auto r=std::make_shared<gumbel::Node>(),a=std::make_shared<gumbel::Node>(),b=std::make_shared<gumbel::Node>(),c=std::make_shared<gumbel::Node>();
  r->player=0;a->player=b->player=1;c->player=0;r->expanded=a->expanded=b->expanded=c->expanded=true;c->n=1;c->q=.2;
  for(auto* x:{r.get()})for(auto child:{a,b}){gumbel::Edge e;e.write().child=child;x->edges.push_back(std::move(e));}
  for(auto x:{a,b}){gumbel::Edge e;e.write().child=c;e.write().visits=1;x->edges.push_back(std::move(e));c->parents.push_back(x);t.refresh(*x);}
  a->parents.push_back(r);b->parents.push_back(r);
  assert(std::abs(b->q+.1)<1e-12);
  c->exact_winner=0;c->distance=1;
  gumbel::Path p;p.leaf=c.get();p.edges={{r.get(),0},{a.get(),0}};for(auto [n,i]:p.edges)++n->edges[i].write().pending;t.backup(p,1);
  // b's only edge leads to the proven win for player 0, so b is proven lost like a.
  assert(c->q==1 && a->q==-1 && b->q==-1 && b->exact_winner==0 && b->edges[0].read().exact_winner==0 && b->edges[0].read().distance==2);}
 // A rarely selected legal edge keeps its weight after eviction, restarts when
 // its retained summary is absent, and receives proofs after reattachment.
 {gumbel::Tree t(3);t.graph=true;t.board.make({0,0});auto p=t.root;p->player=t.board.player;p->expanded=true;p->value=.25;
  auto legal=t.board.legal_moves();for(auto action:legal){gumbel::Edge e;e.action=action;p->edges.push_back(std::move(e));}
  auto child=std::make_shared<gumbel::Node>(p->memory);child->player=p->player;child->n=3;child->q=.5;
  auto& edge=p->edges.back();t.attach(*p,edge,child);edge.write().visits=2;edge.write().sum=1;t.refresh(*p);
  assert(std::abs(p->q-1.25/3)<1e-12 && p->edges.size()==legal.size());
  edge.write().child.reset();t.refresh(*p);assert(std::abs(p->q-1.25/3)<1e-12);
  child=std::make_shared<gumbel::Node>(p->memory);child->player=p->player;t.attach(*p,edge,child);t.refresh(*p);
  assert(edge.read().visits==0 && edge.read().sum==0 && p->q==.25);
  gumbel::Path path;path.leaf=child.get();path.edges={{p.get(),int(p->edges.size()-1)}};edge.write().pending=1;t.backup(path,.8);
  assert(edge.read().visits==1 && edge.read().pending==0 && std::abs(p->q-.525)<1e-12);
  child->exact_winner=1-p->player;child->distance=3;t.propagate(*child,nullptr);
  assert(edge.read().exact_winner==child->exact_winner && edge.read().distance==4 && !edge.read().eligible && p->exact_winner==-1);
  // One touched losing edge cannot exhaust the untouched legal replies.
  assert(std::count_if(p->edges.begin(),p->edges.end(),[](const auto& e){return e.read().eligible;})==int(legal.size())-1);
  for(auto& e:p->edges)if(e.read().exact_winner<0){e.write().exact_winner=child->exact_winner;e.write().distance=2;}
  t.settle(*p);assert(p->exact_winner==child->exact_winner && p->distance==4);
 }
 gumbel::Tree tree(1);tree.advance({0,0});tree.begin(16,4);
 assert(tree.sequence==std::vector<int>({0,0,0,0,1,1,1,1,2,2,3,3,4,4,5,5}));
 gumbel::Node n;n.value=.2;
 for(int i=0;i<3;++i){gumbel::Edge e;e.write().visits=i==0?2:0;e.write().sum=i==0?1.:0;n.edges.push_back(std::move(e));}
 auto q=tree.transformed(n);assert(std::abs(q[0]-5.2)<1e-10 && q[1]==0 && q[2]==0);
 // A proven loss (ineligible, Q -1 with more visits) sets neither the range, the mixed value nor the visit scale.
 {gumbel::Node m;m.value=.2;m.player=0;
  for(int i=0;i<3;++i){gumbel::Edge e;e.write().visits=i<2?2:9;e.write().sum=i==0?1.:-1.;m.edges.push_back(std::move(e));}
  m.edges[2].write().exact_winner=1;m.edges[2].write().eligible=false;
  auto r=tree.transformed(m);assert(std::abs(r[0]-5.2)<1e-10 && std::abs(r[1])<1e-12);}
 // Final selection excludes eliminated actions despite a larger stale score.
 tree.root->expanded=true;tree.root->value=0;
 for(int i=0;i<2;++i){gumbel::Edge e;e.action={i,1};e.logit=i?0:100;tree.root->edges.push_back(std::move(e));}
 tree.prepare_root();tree.root_edges[0].epoch=1;tree.root_edges[1].epoch=4;
 int64_t actions[4];int visits[2];double values[2],scores[2];
 hxg_stats(&tree,actions,visits,values,scores);
 assert(!std::isfinite(scores[0]) && std::isfinite(scores[1]));
 // Reused estimates must not change the initial policy Gumbel sample.
 tree.root->edges[0].logit=1;tree.root_edges[0].gumbel=0;tree.root->edges[0].write().visits=10000;tree.root->edges[0].write().sum=-10000;
 tree.root->edges[1].logit=0;tree.root_edges[1].gumbel=0;tree.root->edges[1].write().visits=10000;tree.root->edges[1].write().sum=10000;
 for(auto& e:tree.root_edges)e.epoch=0;
 int request=tree.request();assert(request>0 && tree.requests.at(request).edges.front().second==0);tree.cancel();
 // Returning after advance resumes the source comparison's priority, bonuses and hold as well as its credits.
 {gumbel::Tree view(3);assert(hxg_share(&view,32));view.root_at({{0,0}});view.begin(8,4);
  int id=view.request();assert(id>0);auto legal=view.requests.at(id).legal;
  std::vector<int64_t> cells;for(auto c:legal){cells.push_back(c.q);cells.push_back(c.r);}
  std::vector<double> zeros(legal.size());view.fulfill(id,cells.data(),zeros.data(),zeros.data(),int(legal.size()));
  Cell action=legal.front();view.priority={action};view.defence[action]=2.;view.hold=true;
  view.advance(action);view.root_at({{0,0}});
  assert(view.hold && view.priority==std::vector<Cell>{action} && view.defence.at(action)==2. && view.budget==8);
 }
 std::cout<<"Exhaustive backup signs and partial proof trees, sequential-halving schedule and mixed-Q transform passed\n";
}
