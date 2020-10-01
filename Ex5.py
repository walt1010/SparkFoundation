{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Importing required packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Importing the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] File SampleSuperstore.csv does not exist: 'SampleSuperstore.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-4-f93e6fb43891>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"SampleSuperstore.csv\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;31m# Assessing the shape of the Data Frame\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mInitial_Shape\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mInitial_Shape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\Users\\tds25\\Anaconda\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36mparser_f\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)\u001b[0m\n\u001b[0;32m    674\u001b[0m         )\n\u001b[0;32m    675\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 676\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    677\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    678\u001b[0m     \u001b[0mparser_f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__name__\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\Users\\tds25\\Anaconda\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    446\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    447\u001b[0m     \u001b[1;31m# Create the parser.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 448\u001b[1;33m     \u001b[0mparser\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfp_or_buf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    449\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    450\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\Users\\tds25\\Anaconda\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m    878\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    879\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 880\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    881\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    882\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\Users\\tds25\\Anaconda\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[1;34m(self, engine)\u001b[0m\n\u001b[0;32m   1112\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mengine\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"c\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1113\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mengine\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"c\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1114\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mCParserWrapper\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1115\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1116\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mengine\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"python\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\Users\\tds25\\Anaconda\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, src, **kwds)\u001b[0m\n\u001b[0;32m   1889\u001b[0m         \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"usecols\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0musecols\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1890\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1891\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_reader\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mparsers\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTextReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1892\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munnamed_cols\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_reader\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munnamed_cols\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1893\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\parsers.pyx\u001b[0m in \u001b[0;36mpandas._libs.parsers.TextReader.__cinit__\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\parsers.pyx\u001b[0m in \u001b[0;36mpandas._libs.parsers.TextReader._setup_parser_source\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] File SampleSuperstore.csv does not exist: 'SampleSuperstore.csv'"
     ]
    }
   ],
   "source": [
    "SuperStoreData = pd.read_csv(\"SampleSuperstore.csv\")\n",
    "SuperStoreData.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Initial Data Inspection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-6-5adcf5c32f96>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "SuperStoreData"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-9088c27b6c02>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdescribe\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "SuperStoreData.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperDataStore' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-6ed27d52e82f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperDataStore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperDataStore' is not defined"
     ]
    }
   ],
   "source": [
    "SuperDataStore.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-9-437579f24b0e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minfo\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "SuperStoreData.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Data Wrangling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-11-2be412d087a6>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m#First, dropping the identical rows from the dataframe. It seems safe to do so.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdrop_duplicates\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkeep\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m'first'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0minplace\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "#First, dropping the identical rows from the dataframe. It seems safe to do so.\n",
    "BeforeW_Shape = SuperStoreData.shape\n",
    "SuperStoreData.drop_duplicates(keep= 'first',inplace=True)\n",
    "AfterW_Shape = SuperStoreData.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-12-5adcf5c32f96>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "Rows_dropped = BeforeW_Shape - AfterW_Shape\n",
    "print(Rows_dropped, \" rows dropped from SuperStoreData\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-13-c2c12c7dbfdd>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Check for nulls\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minfo\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "# 5. Initial High Level Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-14-6d4f9c4a448f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Calculating the total sales and profits, using groupby\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mSales_and_Profits\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Segment\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mround\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mSales_and_Profits\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "# Calculating the total sales and profits, using groupby\n",
    "Sales_and_Profits = SuperStoreData.groupby(\"Segment\").sum().iloc[:,[1,-1]].sum()\n",
    "round(Sales_and_Profits,2) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-15-dbb1b14b0a2d>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Calculating which are the top 10 states by sales and profit\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mTop_10_Sales\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"State\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSales\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlargest\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mTop_10_Profits\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"State\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mProfit\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlargest\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "# Calculating which are the top 10 states by sales and profit\n",
    "Top_10_Sales = SuperStoreData.groupby(\"State\").Sales.sum().nlargest(n =10)\n",
    "Top_10_Profits = SuperStoreData.groupby(\"State\").Profit.sum().nlargest(n =10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Top_10_Sales' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-16-24e9d3c23a65>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Plotting T%op 10 States by Sales\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstyle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0muse\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'seaborn'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mTop_10_Sales\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkind\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m'bar'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"States\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Total Sales\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'Top_10_Sales' is not defined"
     ]
    }
   ],
   "source": [
    "# Plotting Top 10 States by Sales\n",
    "plt.style.use('seaborn')\n",
    "Top_10_Sales.plot(kind ='bar', figsize =(14,8), fontsize =14)\n",
    "plt.xlabel(\"States\", fontsize =13)\n",
    "plt.ylabel(\"Total Sales\",fontsize =13)\n",
    "plt.title(\"Top 10 States by Sales\",fontsize =16)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Top_10_Profits' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-17-42fc0b348314>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Plotting Top 10 States by Sales\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstyle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0muse\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'seaborn'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mTop_10_Profits\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkind\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m'bar'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"States\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Total Profits\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'Top_10_Profits' is not defined"
     ]
    }
   ],
   "source": [
    "# Plotting Top 10 States by Sales\n",
    "plt.style.use('seaborn')\n",
    "Top_10_Profits.plot(kind ='bar', figsize =(14,8), fontsize =14)\n",
    "plt.xlabel(\"States\", fontsize =13)\n",
    "plt.ylabel(\"Total Profits\",fontsize =13)\n",
    "plt.title(\"Top 10 States by Profits\",fontsize =16)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# California has by far the highest sales revenues, but is nearly matched in profits by New York"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-19-6768231c71a6>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Displaying the relationships between Sales, Profit & Discount by scatterplot\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstyle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0muse\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'seaborn'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkind\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"scatter\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m15\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Sales\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m\"Profit\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mc\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m\"Discount\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m20\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmarker\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m\"x\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcolormap\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Total Profits\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Interdependence of Sales, Profit, and Discount\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m\u001b[1;36m16\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "# displaying the correlation between Sales, Profit and Discount\n",
    "correlation=financial.corr()\n",
    "sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interesting result: whilst sales and profot are positively correlated as expected, profits and discoutns are negatively\n",
    "# correlated. Note that this os overall."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Investigating where discounts have had a material negative impact on profits:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'data1' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-22-638a60cbb228>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpivot\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpivot_table\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Sub-Category'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Profit'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mpivot\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkind\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'bar'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'data1' is not defined"
     ]
    }
   ],
   "source": [
    "pivot=pd.pivot_table(data1,index='Sub-Category',values='Sales')\n",
    "pivot.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Largest sales were copiers and machines"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'data1' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-24-638a60cbb228>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpivot\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpivot_table\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Sub-Category'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Profit'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mpivot\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkind\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'bar'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'data1' is not defined"
     ]
    }
   ],
   "source": [
    "pivot=pd.pivot_table(data1,index='Sub-Category',values='Profit')\n",
    "pivot.plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Interestingsly, Copiers had the largest profit, and Machines had the greatest loss!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Profit and Sales analysis\n",
    "# By Segment:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-27-c10cf82d74ec>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m4\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfont_scale\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Region\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Profit\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mhue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Segment\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (12,4))\n",
    "sns.set(font_scale=1, palette= \"viridis\")\n",
    "sns.barplot(data = SuperStoreData , x = \"Region\",y = \"Profit\" ,hue = \"Segment\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# All the segments are approximately equally profitable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Profit and Sales analysis\n",
    "# By Region:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-30-6347d0542ecc>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m4\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfont_scale\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSuperStoreData\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Region\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Profit\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mhue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Category\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (12,4))\n",
    "sns.set(font_scale=1, palette= \"viridis\")\n",
    "sns.barplot(data = SuperStoreData , x = \"Region\",y = \"Profit\" ,hue = \"Category\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Technology is by far the most profitable product set across all regions\n",
    "# Central Region's furniture and office supplies product sets are markedly less profitable than in the other regions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'SuperStoreData' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-32-14813a791cdd>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# \"Slicing\" Furniture Data from rest of the SuperStore data set\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mgb_Category_Furniture\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mSuperStoreData\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Region\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgroupby\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Category\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;31m# Calculating the Correlation matrix Heat Map to identify key factors influening profits\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'SuperStoreData' is not defined"
     ]
    }
   ],
   "source": [
    "# \"Slicing\" Furniture Data from rest of the SuperStore data set\n",
    "gb_Category_Furniture =list(list(SuperStoreData.groupby(\"Region\"))[0][1].groupby(\"Category\"))[0][1]\n",
    "\n",
    "# Calculating the Correlation matrix Heat Map to identify key factors influening profits\n",
    "plt.figure(figsize = (12,8))\n",
    "sns.set(font_scale=1.4)\n",
    "sns.heatmap(gb_Category_Furniture.corr() , annot = True, cmap =\"Reds\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# As before, profitand discounts appear negatively correlated. \n",
    "# The high correlation between postal code and discount may mean that certain states are giving higher discounts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'gb_Category_Furniture' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-34-8ad43362ec6e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfont_scale\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgb_Category_Furniture\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"State\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Profit\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mhue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Sub-Category\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Investigation of Central Region Furniture Category: Profit Analysis(by Sub Category)\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'gb_Category_Furniture' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (12,8))\n",
    "sns.set(font_scale=1, palette= \"viridis\")\n",
    "sns.barplot(data = gb_Category_Furniture , x = \"State\",y = \"Profit\" ,hue = \"Sub-Category\")\n",
    "plt.title(\"Central Region Furniture Category: Profit Analysis(by Sub Category)\", fontsize = 20)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Illinois and Texas stand out in the Central Region as states with the only losses in Furniture, in particular tables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'gb_Category_Furniture' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-36-3ae7c4a8ac12>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfont_scale\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgb_Category_Furniture\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"State\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Profit\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mhue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Segment\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Investigation of Central Region Furniture Category: Profit Analysis(by Segment)\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'gb_Category_Furniture' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (12,8))\n",
    "sns.set(font_scale=1, palette= \"viridis\")\n",
    "sns.barplot(data = gb_Category_Furniture , x = \"State\",y = \"Profit\" ,hue = \"Segment\")\n",
    "plt.title(\"Investigation of Central Region Furniture Category: Profit Analysis(by Segment)\", fontsize = 20)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'gb_Category_Furniture' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-37-1da27d3b0e32>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m8\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfont_scale\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"viridis\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbarplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgb_Category_Furniture\u001b[0m \u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"State\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Discount\"\u001b[0m \u001b[1;33m,\u001b[0m\u001b[0mhue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"Sub-Category\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Furniture Discounts granted per state\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'gb_Category_Furniture' is not defined"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Investigating the discount practises of the states compared to the rest, within Furniture:\n",
    "\n",
    "plt.figure(figsize = (12,8))\n",
    "sns.set(font_scale=1, palette= \"viridis\")\n",
    "sns.barplot(data = gb_Category_Furniture , x = \"State\",y = \"Discount\" ,hue = \"Sub-Category\")\n",
    "plt.title(\"Furniture Discounts granted per state\", fontsize = 20)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Texas and Illinois are the sole discount granters in furniture, amongst the Central Region states\n",
    "# Their discounts range from \n",
    "#                         60% on Furnishings by both states\n",
    "#                         30% on Bookcases and Chairs by both states\n",
    "#                         50% discount on Tables (Illinois) and 30% (Texas)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Major insights thus far:\n",
    "# 1. More Discount is given in the loss making Product sets (Furniture) compared to the profit market product sets\n",
    "# 2. These discounts and hence losses are not uniformly spread, but are concentrated within Central Region\n",
    "# 3. Even within Central region the dscounts an dlosses are highly specific to ony 2 states, Texas and Illinois\n",
    "# 4. Within Furniture, in Texas and Illionis, there is a uniformly high level of discount on furnishings (60%), as well as on \n",
    "# 5. Bookcases and chairs (30%). However the sount s on Tables varies between the 2 states."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
