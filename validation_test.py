from plot_validation import Parser
import unittest

class ValidationTest(unittest.TestCase): 
    files_to_parse = [
        "version1_output/console_output.txt",
        "version1_output/console_output2.txt"
    ] 

    files_to_parse2 = [
        "version2_output/tensorflow-logs.log",
        "version2_output/tensorflow-logs2.log"
    ]
    
    def test_contents1(self):
        parser = Parser(self.files_to_parse, [], [])
        parser.update_points()
        self.assertEqual(parser.steps[0], 667)
        self.assertAlmostEqual(parser.loss[0], 15.336046)
        self.assertEqual(parser.steps[-1], 35000)
        self.assertAlmostEqual(parser.loss[-1], 3.8847911)
        self.assertEqual(len(parser.loss), 56)
        self.assertEqual(len(parser.steps), 56)
    
    def test_contents2(self):
        parser = Parser(self.files_to_parse2, [], [])
        parser.update_points()
        self.assertEqual(parser.steps[0], 1239)
        self.assertAlmostEqual(parser.loss[0], 7.200801)
        self.assertEqual(parser.steps[1], 2309)
        self.assertAlmostEqual(parser.loss[1], 5.7111454)
        self.assertEqual(parser.steps[-1], 60000)
        self.assertAlmostEqual(parser.loss[-1], 2.6621041)
        self.assertEqual(len(parser.loss), 46)
        self.assertEqual(len(parser.steps), 46)


if __name__ == '__main__':
    unittest.main()