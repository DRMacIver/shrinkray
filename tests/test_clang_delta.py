import os

import pytest

from shrinkray.passes.clangdelta import (
    TRANSFORMATIONS,
    ClangDelta,
    clang_delta_pump,
    find_clang_delta,
)
from shrinkray.problem import BasicReductionProblem, WorkContext

BAD_HELLO = b'namespace\x00std{inline\x00namespace\x00__1{template<class>struct\x00char_traits;}}typedef\x00long\x00D;namespace\x00std{namespace\x00__1{template<class\x00_Bhar$,class=char_traits<_Bhar$>>class\x00basic_streambuf;template<class\x00_Bhar$,class=char_traits<_Bhar$>>class\x00C;typedef\x00C<char>E;}D\x00__constexpr_strlen(const\x00char*__str){return\x00__builtin_strlen(__str);}namespace\x00__1{template<>struct\x00char_traits<char>{using\x00C=char;static\x00C\x00$(const\x00C*__s){return\x00__constexpr_strlen(__s);}};template<class\x00_Bhar$>class\x00${public:typedef\x00basic_streambuf<_Bhar$>D;typedef\x00C<_Bhar$>C;D*__sbuf_;$(C&__s):__sbuf_(__s.E()){}$\x00operator=(_Bhar$\x00__c){__sbuf_->sputc(__c);}};class\x00A{public:typedef\x00D\x00$;typedef\x00D\x00C;virtual~A();void*E()const{return\x00__rdbuf_;}$\x00A;C\x00__;C\x00D;C\x00_;void*__rdbuf_;};template<class\x00_Bhar$,class\x00_$1>class\x00B:public\x00A{public:typedef\x00_Bhar$\x00$;typedef\x00_$1\x00C;basic_streambuf<$,C>*E()const;$\x00D()const;};template<class\x00_Bhar$,class\x00_$1>basic_streambuf<_Bhar$,_$1>*B<_Bhar$,_$1>::E()const{return\x00static_cast<basic_streambuf<char>*>(A::E());}template<class\x00_Bhar$,class\x00_$1>_Bhar$\x00B<_Bhar$,_$1>::D()const{}template<class,class>class\x00basic_streambuf{public:int\x00sputc(char);};}template<class\x00_Bhar$,class\x00_$1$1>_$1$1\x00__pad_and_output(_$1$1\x00__s,const\x00_Bhar$*__ob,const\x00_Bhar$*__oe){for(;__ob<__oe;++__ob)__s=*__ob;}namespace\x00__1{template<class\x00_Bhar$,class\x00_$1>class\x00C:virtual\x00public\x00B<char,_$1>{};template<class\x00_Bhar$,class\x00_$1>C<_Bhar$>&__put_character_se$1(C<_$1>&__os,const\x00_Bhar$*__str,D\x00__len){try{typedef\x00$<_Bhar$>_$A;__pad_and_output(_$A(__os),__str,__str+__len);}catch(...){}return\x00__os;}template<class\x00_Bhar$,class\x00_$1>C<_Bhar$>&operator<<(C<_Bhar$,_$1>&__os,const\x00_Bhar$*__str){return\x00__put_character_se$1(__os,__str,_$1::$(__str));}extern\x00E\x00cout;}}int\x00main(){std::cout<<"Hello";}'
CRASHER = b'namespace{\n    inline namespace __1{\n        template<class>struct __attribute(())char_traits;\n        template<class _C,class=char_traits<_C>>class __attribute(())C;\n    }\n}\nnamespace{\n    namespace __1{\n        class __attribute(())ios_base{\n            public:~ios_base();\n        }\n        ;\n        template<class,class>class __attribute(())D:ios_base{}\n        ;\n        template<class _C,class _s>class __attribute(())C:D<_C,_s>{\n            public:void operator<<(C&(C&));\n        }\n        ;\n        template<class _C,class _s>__attribute(())C<int>operator<<(C<_s>,_C){}\n        template<class _C,class _s>C<_C>&endl(C<_s>&);\n    }\n}\nnamespace{\n    extern __attribute(())C<char>cout;\n}\nint main(){\n    cout<<""<<endl;\n}\n'


@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
@pytest.mark.parametrize("source", [BAD_HELLO, CRASHER])
async def test_can_apply_transformations(transformation, source):
    cd_exec = find_clang_delta()
    assert os.path.exists(cd_exec)
    cd = ClangDelta(cd_exec)
    pump = clang_delta_pump(cd, transformation)

    async def equals_source(x):
        return x == source

    problem = BasicReductionProblem(
        source, equals_source, work=WorkContext(parallelism=1)
    )

    await pump(problem)
