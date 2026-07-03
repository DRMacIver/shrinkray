  template < class b, b...  >  b operator""c();
        template < class = decltype(""c)> void d(int ) {
         d(2)