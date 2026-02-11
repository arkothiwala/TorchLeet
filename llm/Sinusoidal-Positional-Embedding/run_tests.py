import sys
from test_sinusoidal_position_encodings import *
import traceback

def run_tests():
    tests = [
        test_unit_magnitude_check,
        test_dot_product_shift_invariance,
        test_pairwise_dot_product_identity,
        test_linear_shift_operator,
        test_frequency_pair_equality,
        test_constant_norm_per_pair,
        test_orthogonality_across_frequencies,
        test_long_sequence_stability,
        test_batch_consistency,
        test_dtype_consistency
    ]
    
    passed = 0
    failed = 0
    results = []
    
    print("Running Sinusoidal Positional Embedding Tests...")
    print("=" * 50)
    
    for test in tests:
        test_name = test.__name__
        try:
            test()
            print(f"✅ {test_name} PASSED")
            passed += 1
            results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ {test_name} FAILED")
            # print(traceback.format_exc())
            failed += 1
            results.append((test_name, "FAILED", str(e)))
            
    print("=" * 50)
    print(f"Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nFailure Details:")
        for name, status, error in results:
            if status == "FAILED":
                print(f"- {name}: {error}")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
