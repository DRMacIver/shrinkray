class a {   void b(int);   void c(); };
 void a::c() {   auto lam = [&](auto asdf) { b(asdf); };   lam(0); }
