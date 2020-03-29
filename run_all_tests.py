import unittest
import sys
import os


if __name__ == "__main__":
    try:
        dir = os.path.dirname(os.path.abspath(__file__))
        components = [os.path.join(dir, o)
                      for o in os.listdir(dir)
                        if os.path.isdir(os.path.join(dir, o)) and o.startswith("pandas-ml")]

        print("Testing components:")
        print(components)
        tests_run = 0
        tests_failed = 0

        for component in components:
            sys.path.insert(0, component)

        for component in components:
            all_tests = unittest.TestLoader().discover(component, pattern='test_*.py')

            print(f"running tests for {component}")
            test_result = unittest.TextTestRunner().run(all_tests)

            assert len(test_result.errors) + len(test_result.failures) <= 0, f"test failed for {component}"

            tests_run += test_result.testsRun
            tests_failed += len(test_result.failures) + len(test_result.errors)
            print("\n-------------------------------------------------\n")

        print(f"tests run: {tests_run}")

        if tests_run <= 0:
            sys.exit(1)
    except:
        # make sure the CI-Pipeline fails
        sys.exit(1)
