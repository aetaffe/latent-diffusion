import unittest
from cholec8kdataset import Cholec8kDataset

class MyTestCase(unittest.TestCase):
    def test_create_cholec_8k_dataset(self):
        dataset = Cholec8kDataset('/media/alex/1TBSSD/SSD/Segmentation_Datasets/CholecSeg8k')
        self.assertEqual(len(dataset), 8080)
        self.assertEqual(dataset[0].shape, (680, 680, 4))



if __name__ == '__main__':
    unittest.main()
