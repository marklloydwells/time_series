import unittest
import pandas as pd
import numpy as np
from quant_ds_interview_task.app import remove_outliers

class Tests(unittest.TestCase):
    def test_remove_outliers(self):
        df = pd.DataFrame(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13],
                           [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]]).T,
                          index=['2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', 
                                 '2018-01-06', '2018-01-07', '2018-01-08', '2018-01-09',
                                 '2018-01-10', '2018-01-11', '2018-01-12', '2018-01-13',
                                 '2018-01-14',
                                 ],
                          columns=['EUR/USD', 'USD/JPY'],
                          )
        
        sn = pd.DataFrame(np.array([[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6],
                           [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]]).T,
                          index=['2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', 
                                 '2018-01-06', '2018-01-07', '2018-01-08', '2018-01-09',
                                 '2018-01-10', '2018-01-11', '2018-01-12', '2018-01-13',
                                 '2018-01-14',
                                 ],
                          columns=['EUR/USD', 'USD/JPY'],
                          )
        
        expected = pd.DataFrame(np.array([[np.nan,np.nan,3,4,5,6,7,8,9,10,11,11,11],
                           [np.nan,np.nan,-4,-3,-2,-1,0,1,2,3,4,4,4]]).T,
                          index=['2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', 
                                 '2018-01-06', '2018-01-07', '2018-01-08', '2018-01-09',
                                 '2018-01-10', '2018-01-11', '2018-01-12', '2018-01-13',
                                 '2018-01-14',
                                 ],
                          columns=['EUR/USD', 'USD/JPY'],
                          )
        
        
        self.assertTrue(remove_outliers(df, sn,).equals(expected))
        

    
if __name__ == "__main__":
    t = Tests()
    t.test_remove_outliers()