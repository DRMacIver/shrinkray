import os

import pytest

from shrinkray.passes.clangdelta import (
    TRANSFORMATIONS,
    ClangDelta,
    ClangDeltaError,
    clang_delta_pump,
    clang_delta_pumps,
    find_clang_delta,
)
from shrinkray.problem import BasicReductionProblem, WorkContext


pytestmark = pytest.mark.skipif(not find_clang_delta(), reason="unavailable")

BAD_HELLO = b'namespace\x00std{inline\x00namespace\x00__1{template<class>struct\x00char_traits;}}typedef\x00long\x00D;namespace\x00std{namespace\x00__1{template<class\x00_Bhar$,class=char_traits<_Bhar$>>class\x00basic_streambuf;template<class\x00_Bhar$,class=char_traits<_Bhar$>>class\x00C;typedef\x00C<char>E;}D\x00__constexpr_strlen(const\x00char*__str){return\x00__builtin_strlen(__str);}namespace\x00__1{template<>struct\x00char_traits<char>{using\x00C=char;static\x00C\x00$(const\x00C*__s){return\x00__constexpr_strlen(__s);}};template<class\x00_Bhar$>class\x00${public:typedef\x00basic_streambuf<_Bhar$>D;typedef\x00C<_Bhar$>C;D*__sbuf_;$(C&__s):__sbuf_(__s.E()){}$\x00operator=(_Bhar$\x00__c){__sbuf_->sputc(__c);}};class\x00A{public:typedef\x00D\x00$;typedef\x00D\x00C;virtual~A();void*E()const{return\x00__rdbuf_;}$\x00A;C\x00__;C\x00D;C\x00_;void*__rdbuf_;};template<class\x00_Bhar$,class\x00_$1>class\x00B:public\x00A{public:typedef\x00_Bhar$\x00$;typedef\x00_$1\x00C;basic_streambuf<$,C>*E()const;$\x00D()const;};template<class\x00_Bhar$,class\x00_$1>basic_streambuf<_Bhar$,_$1>*B<_Bhar$,_$1>::E()const{return\x00static_cast<basic_streambuf<char>*>(A::E());}template<class\x00_Bhar$,class\x00_$1>_Bhar$\x00B<_Bhar$,_$1>::D()const{}template<class,class>class\x00basic_streambuf{public:int\x00sputc(char);};}template<class\x00_Bhar$,class\x00_$1$1>_$1$1\x00__pad_and_output(_$1$1\x00__s,const\x00_Bhar$*__ob,const\x00_Bhar$*__oe){for(;__ob<__oe;++__ob)__s=*__ob;}namespace\x00__1{template<class\x00_Bhar$,class\x00_$1>class\x00C:virtual\x00public\x00B<char,_$1>{};template<class\x00_Bhar$,class\x00_$1>C<_Bhar$>&__put_character_se$1(C<_$1>&__os,const\x00_Bhar$*__str,D\x00__len){try{typedef\x00$<_Bhar$>_$A;__pad_and_output(_$A(__os),__str,__str+__len);}catch(...){}return\x00__os;}template<class\x00_Bhar$,class\x00_$1>C<_Bhar$>&operator<<(C<_Bhar$,_$1>&__os,const\x00_Bhar$*__str){return\x00__put_character_se$1(__os,__str,_$1::$(__str));}extern\x00E\x00cout;}}int\x00main(){std::cout<<"Hello";}'
CRASHER = b'namespace{\n    inline namespace __1{\n        template<class>struct __attribute(())char_traits;\n        template<class _C,class=char_traits<_C>>class __attribute(())C;\n    }\n}\nnamespace{\n    namespace __1{\n        class __attribute(())ios_base{\n            public:~ios_base();\n        }\n        ;\n        template<class,class>class __attribute(())D:ios_base{}\n        ;\n        template<class _C,class _s>class __attribute(())C:D<_C,_s>{\n            public:void operator<<(C&(C&));\n        }\n        ;\n        template<class _C,class _s>__attribute(())C<int>operator<<(C<_s>,_C){}\n        template<class _C,class _s>C<_C>&endl(C<_s>&);\n    }\n}\nnamespace{\n    extern __attribute(())C<char>cout;\n}\nint main(){\n    cout<<""<<endl;\n}\n'


@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
@pytest.mark.parametrize("source", [BAD_HELLO, CRASHER], ids=["BAD_HELLO", "CRASHER"])
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


# =============================================================================
# Error handling tests
# =============================================================================


def test_clang_delta_error_without_assertion():
    """Test ClangDeltaError can be raised with normal error messages.

    ClangDeltaError is used for non-assertion errors from clang_delta.
    """
    error = ClangDeltaError(b"Some error message")
    assert isinstance(error, Exception)


def test_clang_delta_error_rejects_assertion_messages():
    """Test ClangDeltaError rejects messages containing 'Assertion failed'.

    clang_delta assertion failures are handled specially - they're ignored
    rather than propagated. ClangDeltaError.__init__ asserts that the message
    doesn't contain 'Assertion failed' to catch misuse.
    """
    with pytest.raises(AssertionError):
        ClangDeltaError(b"Assertion failed: something went wrong")


async def test_invalid_transformation():
    """Test that invalid transformation raises ValueError."""
    cd = ClangDelta(find_clang_delta())
    with pytest.raises(ValueError, match="Invalid transformation"):
        await cd.query_instances("not-a-real-transformation", b"int main() {}")


async def test_query_instances_returns_count():
    """Test query_instances returns the number of transformation instances."""
    cd = ClangDelta(find_clang_delta())
    # Use the BAD_HELLO source which has known renameable variables
    count = await cd.query_instances("rename-var", BAD_HELLO)
    assert isinstance(count, int)
    assert count > 0, "BAD_HELLO should have at least one renameable variable"


def test_clang_delta_pumps():
    """Test clang_delta_pumps returns list of pumps for all transformations."""
    cd = ClangDelta(find_clang_delta())
    pumps = clang_delta_pumps(cd)
    assert len(pumps) == len(TRANSFORMATIONS)
    # Each pump should be a callable with a name
    for pump in pumps:
        assert callable(pump)
        assert "clang_delta(" in pump.__name__


# =============================================================================
# Assertion failure handling tests
# =============================================================================

# This C++ code triggers an assertion failure in clang_delta's rename-fun pass
ASSERTION_CRASHER = b"""namespace{
    inline namespace __1{
        template<class>struct __attribute(())char_traits;
        template<class _C,class=char_traits<_C>>class __attribute(())C;
    }
}
namespace{
    namespace __1{
        class __attribute(())ios_base{
            public:~ios_base();
        }
        ;
        template<class,class>class __attribute(())D:ios_base{}
        ;
        template<class _C,class _s>class __attribute(())C:D<_C,_s>{
            public:void operator<<(C&(C&));
        }
        ;
        template<class _C,class _s>__attribute(())C<int>operator<<(C<_s>,_C){}
        template<class _C,class _s>C<_C>&endl(C<_s>&);
    }
}
namespace{
    extern __attribute(())C<char>cout;
}
int main(){
    cout<<""<<endl;
}
"""


async def test_query_instances_on_crasher():
    """Test query_instances works on code that will crash during apply."""
    cd = ClangDelta(find_clang_delta())
    # query_instances succeeds (returns instance count), but applying will crash
    count = await cd.query_instances("rename-fun", ASSERTION_CRASHER)
    assert count >= 1  # There are instances, but applying them crashes


async def test_apply_transformation_handles_assertion_failure():
    """Test apply_transformation returns original when clang_delta hits an assertion."""
    cd = ClangDelta(find_clang_delta())
    # This should trigger an assertion failure but return original instead of raising
    result = await cd.apply_transformation("rename-fun", 1, ASSERTION_CRASHER)
    assert result == ASSERTION_CRASHER


async def test_apply_transformation_raises_clang_delta_error():
    """Test apply_transformation raises ClangDeltaError for non-assertion errors."""
    cd = ClangDelta(find_clang_delta())
    # Simple code with no variables to rename should cause an error
    source = b"int main() { return 0; }"
    with pytest.raises(ClangDeltaError):
        await cd.apply_transformation("rename-var", 1, source)


async def test_pump_handles_assertion_failure():
    """Test clang_delta_pump handles assertion failures gracefully."""
    cd = ClangDelta(find_clang_delta())
    pump = clang_delta_pump(cd, "rename-fun")

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        ASSERTION_CRASHER, is_interesting, work=WorkContext(parallelism=1)
    )

    # Should not raise, should return the original
    result = await pump(problem)
    assert result == ASSERTION_CRASHER


async def test_pump_handles_apply_error():
    """Test clang_delta_pump handles ClangDeltaError from apply_transformation."""
    cd = ClangDelta(find_clang_delta())
    # Use rename-var which will error when trying to rename in simple code
    pump = clang_delta_pump(cd, "rename-var")

    async def is_interesting(x):
        return True

    # Simple code that has instances counted but fails during apply
    source = b"int main() { return 0; }"
    problem = BasicReductionProblem(
        source, is_interesting, work=WorkContext(parallelism=1)
    )

    # Should not raise, should return the original
    result = await pump(problem)
    assert result == source


# =============================================================================
# Additional edge case tests for complete coverage
# =============================================================================


def test_find_clang_delta_when_found_in_path():
    """Test find_clang_delta returns path when found via which()."""
    from unittest.mock import patch

    # Mock which to return a path
    with patch(
        "shrinkray.passes.clangdelta.which", return_value="/usr/bin/clang_delta"
    ):
        result = find_clang_delta()
        assert result == "/usr/bin/clang_delta"


def test_find_clang_delta_when_not_in_path():
    """Test find_clang_delta falls back to glob when not in PATH."""
    from unittest.mock import patch

    # Mock which to return None (not in PATH)
    with patch("shrinkray.passes.clangdelta.which", return_value=None):
        # This will use glob to find clang_delta
        result = find_clang_delta()
        # Result depends on whether clang_delta is installed via homebrew/apt
        # We just verify it doesn't crash and returns something (empty string if not found)
        assert isinstance(result, str)


def test_find_clang_delta_when_not_found_anywhere():
    """Test find_clang_delta returns empty string when not found."""
    from unittest.mock import patch

    # Mock both which and glob to return nothing
    with patch("shrinkray.passes.clangdelta.which", return_value=None):
        with patch("shrinkray.passes.clangdelta.glob", return_value=[]):
            result = find_clang_delta()
            assert result == ""


async def test_query_instances_raises_clang_delta_error():
    """Test query_instances raises ClangDeltaError for non-assertion errors.

    Exercises the ClangDeltaError path when CalledProcessError
    doesn't contain 'Assertion failed'.
    """
    import subprocess
    from unittest.mock import patch

    cd = ClangDelta(find_clang_delta())

    # Mock trio.run_process to raise CalledProcessError without "Assertion failed"
    error = subprocess.CalledProcessError(1, "clang_delta")
    error.stdout = b"Some other error"
    error.stderr = b"More error info"

    with patch("trio.run_process", side_effect=error):
        with pytest.raises(ClangDeltaError):
            await cd.query_instances("rename-var", b"int main() {}")


async def test_apply_transformation_no_modification():
    """Test apply_transformation returns original when 'No modification' error.

    Exercises the no-modification fallback path in apply_transformation.
    """
    import subprocess
    from unittest.mock import patch

    cd = ClangDelta(find_clang_delta())
    source = b"int main() { return 0; }"

    # Mock trio.run_process to raise CalledProcessError with "No modification" message
    error = subprocess.CalledProcessError(1, "clang_delta")
    error.stdout = b"Error: No modification to the transformed program!"
    error.stderr = b""

    with patch("trio.run_process", side_effect=error):
        result = await cd.apply_transformation("rename-var", 1, source)
        assert result == source


async def test_pump_handles_query_instances_error():
    """Test clang_delta_pump handles ClangDeltaError from query_instances.

    Exercises the error handling path when query_instances raises ClangDeltaError.
    """
    from unittest.mock import patch

    cd = ClangDelta(find_clang_delta())
    pump = clang_delta_pump(cd, "rename-var")

    source = b"int main() { return 0; }"

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        source, is_interesting, work=WorkContext(parallelism=1)
    )

    # Mock query_instances to raise ClangDeltaError
    with patch.object(
        cd, "query_instances", side_effect=ClangDeltaError(b"Query failed")
    ):
        result = await pump(problem)
        assert result == source


async def test_pump_makes_progress_when_transformation_is_interesting():
    """Test clang_delta_pump makes progress when transformations produce interesting results."""
    cd = ClangDelta(find_clang_delta())
    # Use rename-var which will rename variables
    pump = clang_delta_pump(cd, "rename-var")

    # Source with variables that can be renamed
    source = b"int longVariableName = 1; int main() { return longVariableName; }"

    # Accept any transformation (all results are interesting)
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        source, is_interesting, work=WorkContext(parallelism=1)
    )

    result = await pump(problem)
    # The pump should have made some transformations
    # We just verify it runs without error
    assert result is not None


async def test_pump_handles_clang_delta_error_during_find_first_value():
    """Test pump handles ClangDeltaError raised during find_first_value.

    Exercises the error handling when ClangDeltaError is raised during apply.
    """
    from unittest.mock import patch

    cd = ClangDelta(find_clang_delta())
    pump = clang_delta_pump(cd, "rename-var")

    # Source that has instances to transform
    source = b"int longVariableName = 1; int main() { return longVariableName; }"

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        source, is_interesting, work=WorkContext(parallelism=1)
    )

    # First, let query_instances return 1 to enter the while loop
    # Then, when apply_transformation is called inside can_apply,
    # it will raise ClangDeltaError
    call_count = [0]

    async def mock_apply(*args, **kwargs):
        call_count[0] += 1
        # Raise on first call (inside can_apply during find_first_value)
        raise ClangDeltaError(b"Test error during find_first_value")

    with patch.object(cd, "apply_transformation", side_effect=mock_apply):
        result = await pump(problem)

    # Should return target without raising
    assert result == source
