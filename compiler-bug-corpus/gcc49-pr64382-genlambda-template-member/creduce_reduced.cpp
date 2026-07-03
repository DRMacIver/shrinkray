template < typename > struct Queue {
  void push(int) {
    auto L = [=](auto v) { push(v); };
    traverse(L);
  }
  template < typename F > void traverse(F f) { f(int()); }
};
template struct Queue< int >;
