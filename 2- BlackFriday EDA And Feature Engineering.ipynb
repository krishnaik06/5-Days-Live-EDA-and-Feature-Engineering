{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1b8d424a",
   "metadata": {},
   "source": [
    "## Black Friday Dataset EDA And Feature Engineering\n",
    "### Cleaning and preparing the data for model training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c11b3cde",
   "metadata": {},
   "outputs": [],
   "source": [
    "## dataset link: https://www.kaggle.com/sdolezel/black-friday?select=train.csv\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc4411e3",
   "metadata": {},
   "source": [
    "# Problem Statement\n",
    "A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.\n",
    "The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.\n",
    "\n",
    "Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "68d3b6aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN      8370  \n",
       "1                 6.0                14.0     15200  \n",
       "2                 NaN                 NaN      1422  \n",
       "3                14.0                 NaN      1057  \n",
       "4                 NaN                 NaN      7969  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#importing the dataset\n",
    "df_train=pd.read_csv('blackFriday_train.csv')\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "23a546ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000004</td>\n",
       "      <td>P00128942</td>\n",
       "      <td>M</td>\n",
       "      <td>46-50</td>\n",
       "      <td>7</td>\n",
       "      <td>B</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000009</td>\n",
       "      <td>P00113442</td>\n",
       "      <td>M</td>\n",
       "      <td>26-35</td>\n",
       "      <td>17</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000010</td>\n",
       "      <td>P00288442</td>\n",
       "      <td>F</td>\n",
       "      <td>36-45</td>\n",
       "      <td>1</td>\n",
       "      <td>B</td>\n",
       "      <td>4+</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000010</td>\n",
       "      <td>P00145342</td>\n",
       "      <td>F</td>\n",
       "      <td>36-45</td>\n",
       "      <td>1</td>\n",
       "      <td>B</td>\n",
       "      <td>4+</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000011</td>\n",
       "      <td>P00053842</td>\n",
       "      <td>F</td>\n",
       "      <td>26-35</td>\n",
       "      <td>1</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>5.0</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender    Age  Occupation City_Category  \\\n",
       "0  1000004  P00128942      M  46-50           7             B   \n",
       "1  1000009  P00113442      M  26-35          17             C   \n",
       "2  1000010  P00288442      F  36-45           1             B   \n",
       "3  1000010  P00145342      F  36-45           1             B   \n",
       "4  1000011  P00053842      F  26-35           1             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               1                   1   \n",
       "1                          0               0                   3   \n",
       "2                         4+               1                   5   \n",
       "3                         4+               1                   4   \n",
       "4                          1               0                   4   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  \n",
       "0                11.0                 NaN  \n",
       "1                 5.0                 NaN  \n",
       "2                14.0                 NaN  \n",
       "3                 9.0                 NaN  \n",
       "4                 5.0                12.0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##  import the test data\n",
    "df_test=pd.read_csv('blackFriday_test.csv')\n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b5a29311",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN    8370.0  \n",
       "1                 6.0                14.0   15200.0  \n",
       "2                 NaN                 NaN    1422.0  \n",
       "3                14.0                 NaN    1057.0  \n",
       "4                 NaN                 NaN    7969.0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##MErge both train and test data\n",
    "df=df_train.append(df_test)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "663221b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 783667 entries, 0 to 233598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   User_ID                     783667 non-null  int64  \n",
      " 1   Product_ID                  783667 non-null  object \n",
      " 2   Gender                      783667 non-null  object \n",
      " 3   Age                         783667 non-null  object \n",
      " 4   Occupation                  783667 non-null  int64  \n",
      " 5   City_Category               783667 non-null  object \n",
      " 6   Stay_In_Current_City_Years  783667 non-null  object \n",
      " 7   Marital_Status              783667 non-null  int64  \n",
      " 8   Product_Category_1          783667 non-null  int64  \n",
      " 9   Product_Category_2          537685 non-null  float64\n",
      " 10  Product_Category_3          237858 non-null  float64\n",
      " 11  Purchase                    550068 non-null  float64\n",
      "dtypes: float64(3), int64(4), object(5)\n",
      "memory usage: 77.7+ MB\n"
     ]
    }
   ],
   "source": [
    "##Basic \n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "15840fa4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>7.836670e+05</td>\n",
       "      <td>783667.000000</td>\n",
       "      <td>783667.000000</td>\n",
       "      <td>783667.000000</td>\n",
       "      <td>537685.000000</td>\n",
       "      <td>237858.000000</td>\n",
       "      <td>550068.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.003029e+06</td>\n",
       "      <td>8.079300</td>\n",
       "      <td>0.409777</td>\n",
       "      <td>5.366196</td>\n",
       "      <td>9.844506</td>\n",
       "      <td>12.668605</td>\n",
       "      <td>9263.968713</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.727267e+03</td>\n",
       "      <td>6.522206</td>\n",
       "      <td>0.491793</td>\n",
       "      <td>3.878160</td>\n",
       "      <td>5.089093</td>\n",
       "      <td>4.125510</td>\n",
       "      <td>5023.065394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000001e+06</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.001519e+06</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>5823.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.003075e+06</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>8047.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.004478e+06</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>12054.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.006040e+06</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>23961.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            User_ID     Occupation  Marital_Status  Product_Category_1  \\\n",
       "count  7.836670e+05  783667.000000   783667.000000       783667.000000   \n",
       "mean   1.003029e+06       8.079300        0.409777            5.366196   \n",
       "std    1.727267e+03       6.522206        0.491793            3.878160   \n",
       "min    1.000001e+06       0.000000        0.000000            1.000000   \n",
       "25%    1.001519e+06       2.000000        0.000000            1.000000   \n",
       "50%    1.003075e+06       7.000000        0.000000            5.000000   \n",
       "75%    1.004478e+06      14.000000        1.000000            8.000000   \n",
       "max    1.006040e+06      20.000000        1.000000           20.000000   \n",
       "\n",
       "       Product_Category_2  Product_Category_3       Purchase  \n",
       "count       537685.000000       237858.000000  550068.000000  \n",
       "mean             9.844506           12.668605    9263.968713  \n",
       "std              5.089093            4.125510    5023.065394  \n",
       "min              2.000000            3.000000      12.000000  \n",
       "25%              5.000000            9.000000    5823.000000  \n",
       "50%              9.000000           14.000000    8047.000000  \n",
       "75%             15.000000           16.000000   12054.000000  \n",
       "max             18.000000           18.000000   23961.000000  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "27ecb6b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['User_ID'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c9e3089b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  P00069042      F  0-17          10             A   \n",
       "1  P00248942      F  0-17          10             A   \n",
       "2  P00087842      F  0-17          10             A   \n",
       "3  P00085442      F  0-17          10             A   \n",
       "4  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN    8370.0  \n",
       "1                 6.0                14.0   15200.0  \n",
       "2                 NaN                 NaN    1422.0  \n",
       "3                14.0                 NaN    1057.0  \n",
       "4                 NaN                 NaN    7969.0  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "488f69d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>M</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233594</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233595</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233596</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233597</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233598</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>783667 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        M\n",
       "0       0\n",
       "1       0\n",
       "2       0\n",
       "3       0\n",
       "4       1\n",
       "...    ..\n",
       "233594  0\n",
       "233595  0\n",
       "233596  0\n",
       "233597  0\n",
       "233598  0\n",
       "\n",
       "[783667 rows x 1 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender']=pd.get_dummies(df['Gender'],drop_first=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "41440b97",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender   Age  Occupation City_Category  \\\n",
       "0  P00069042       0  0-17          10             A   \n",
       "1  P00248942       0  0-17          10             A   \n",
       "2  P00087842       0  0-17          10             A   \n",
       "3  P00085442       0  0-17          10             A   \n",
       "4  P00285442       1   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN    8370.0  \n",
       "1                 6.0                14.0   15200.0  \n",
       "2                 NaN                 NaN    1422.0  \n",
       "3                14.0                 NaN    1057.0  \n",
       "4                 NaN                 NaN    7969.0  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##HAndling categorical feature Gender\n",
    "df['Gender']=df['Gender'].map({'F':0,'M':1})\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d2b84c18",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['0-17', '55+', '26-35', '46-50', '51-55', '36-45', '18-25'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Handle categorical feature Age\n",
    "df['Age'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f2a47d61",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pd.get_dummies(df['Age'],drop_first=True)\n",
    "df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5eebdca4",
   "metadata": {},
   "outputs": [],
   "source": [
    "##second technqiue\n",
    "from sklearn import preprocessing\n",
    " \n",
    "# label_encoder object knows how to understand word labels.\n",
    "label_encoder = preprocessing.LabelEncoder()\n",
    " \n",
    "# Encode labels in column 'species'.\n",
    "df['Age']= label_encoder.fit_transform(df['Age'])\n",
    " \n",
    "df['Age'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ccc535fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation City_Category  \\\n",
       "0  P00069042       0    1          10             A   \n",
       "1  P00248942       0    1          10             A   \n",
       "2  P00087842       0    1          10             A   \n",
       "3  P00085442       0    1          10             A   \n",
       "4  P00285442       1    7          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN    8370.0  \n",
       "1                 6.0                14.0   15200.0  \n",
       "2                 NaN                 NaN    1422.0  \n",
       "3                14.0                 NaN    1057.0  \n",
       "4                 NaN                 NaN    7969.0  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f1acab05",
   "metadata": {},
   "outputs": [],
   "source": [
    "##fixing categorical City_categort\n",
    "df_city=pd.get_dummies(df['City_Category'],drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8b7b3671",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   B  C\n",
       "0  0  0\n",
       "1  0  0\n",
       "2  0  0\n",
       "3  0  0\n",
       "4  0  1"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_city.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "307f8aca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation City_Category  \\\n",
       "0  P00069042       0    1          10             A   \n",
       "1  P00248942       0    1          10             A   \n",
       "2  P00087842       0    1          10             A   \n",
       "3  P00085442       0    1          10             A   \n",
       "4  P00285442       1    7          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  B  C  \n",
       "0                 NaN                 NaN    8370.0  0  0  \n",
       "1                 6.0                14.0   15200.0  0  0  \n",
       "2                 NaN                 NaN    1422.0  0  0  \n",
       "3                14.0                 NaN    1057.0  0  0  \n",
       "4                 NaN                 NaN    7969.0  0  1  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.concat([df,df_city],axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c6a6ed8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "##drop City Category Feature\n",
    "df.drop('City_Category',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "994205dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation Stay_In_Current_City_Years  \\\n",
       "0  P00069042       0    1          10                          2   \n",
       "1  P00248942       0    1          10                          2   \n",
       "2  P00087842       0    1          10                          2   \n",
       "3  P00085442       0    1          10                          2   \n",
       "4  P00285442       1    7          16                         4+   \n",
       "\n",
       "   Marital_Status  Product_Category_1  Product_Category_2  Product_Category_3  \\\n",
       "0               0                   3                 NaN                 NaN   \n",
       "1               0                   1                 6.0                14.0   \n",
       "2               0                  12                 NaN                 NaN   \n",
       "3               0                  12                14.0                 NaN   \n",
       "4               0                   8                 NaN                 NaN   \n",
       "\n",
       "   Purchase  B  C  \n",
       "0    8370.0  0  0  \n",
       "1   15200.0  0  0  \n",
       "2    1422.0  0  0  \n",
       "3    1057.0  0  0  \n",
       "4    7969.0  0  1  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "97a1db89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Product_ID                         0\n",
       "Gender                             0\n",
       "Age                                0\n",
       "Occupation                         0\n",
       "Stay_In_Current_City_Years         0\n",
       "Marital_Status                     0\n",
       "Product_Category_1                 0\n",
       "Product_Category_2            245982\n",
       "Product_Category_3            545809\n",
       "Purchase                      233599\n",
       "B                                  0\n",
       "C                                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Missing Values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3ad28ef5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([nan,  6., 14.,  2.,  8., 15., 16., 11.,  5.,  3.,  4., 12.,  9.,\n",
       "       10., 17., 13.,  7., 18.])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Focus on replacing missing values\n",
    "df['Product_Category_2'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "1fbc9976",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8.0     91317\n",
       "14.0    78834\n",
       "2.0     70498\n",
       "16.0    61687\n",
       "15.0    54114\n",
       "5.0     37165\n",
       "4.0     36705\n",
       "6.0     23575\n",
       "11.0    20230\n",
       "17.0    19104\n",
       "13.0    15054\n",
       "9.0      8177\n",
       "12.0     7801\n",
       "10.0     4420\n",
       "3.0      4123\n",
       "18.0     4027\n",
       "7.0       854\n",
       "Name: Product_Category_2, dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Product_Category_2'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "49c0dd02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8.0"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Product_Category_2'].mode()[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a2312a53",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Replace the missing values with mode\n",
    "df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "8453f1c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Product_Category_2'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1bb99553",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([nan, 14., 17.,  5.,  4., 16., 15.,  8.,  9., 13.,  6., 12.,  3.,\n",
       "       18., 11., 10.])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Product_category 3 replace missing values\n",
    "df['Product_Category_3'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "69c60d56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "16.0    46469\n",
       "15.0    39968\n",
       "14.0    26283\n",
       "17.0    23818\n",
       "5.0     23799\n",
       "8.0     17861\n",
       "9.0     16532\n",
       "12.0    13115\n",
       "13.0     7849\n",
       "6.0      6888\n",
       "18.0     6621\n",
       "4.0      2691\n",
       "11.0     2585\n",
       "10.0     2501\n",
       "3.0       878\n",
       "Name: Product_Category_3, dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Product_Category_3'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "102ee1aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Replace the missing values with mode\n",
    "df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "87cd27df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>8370.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1057.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>7969.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation Stay_In_Current_City_Years  \\\n",
       "0  P00069042       0    1          10                          2   \n",
       "1  P00248942       0    1          10                          2   \n",
       "2  P00087842       0    1          10                          2   \n",
       "3  P00085442       0    1          10                          2   \n",
       "4  P00285442       1    7          16                         4+   \n",
       "\n",
       "   Marital_Status  Product_Category_1  Product_Category_2  Product_Category_3  \\\n",
       "0               0                   3                 8.0                16.0   \n",
       "1               0                   1                 6.0                14.0   \n",
       "2               0                  12                 8.0                16.0   \n",
       "3               0                  12                14.0                16.0   \n",
       "4               0                   8                 8.0                16.0   \n",
       "\n",
       "   Purchase  B  C  \n",
       "0    8370.0  0  0  \n",
       "1   15200.0  0  0  \n",
       "2    1422.0  0  0  \n",
       "3    1057.0  0  0  \n",
       "4    7969.0  0  1  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "d63397f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(783667, 12)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "92e68302",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['2', '4+', '3', '1', '0'], dtype=object)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Stay_In_Current_City_Years'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "76cbc223",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\AppData\\Local\\Temp/ipykernel_24288/2063355665.py:1: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')\n"
     ]
    }
   ],
   "source": [
    "df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "41013bd8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>8370.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1057.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>7969.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation Stay_In_Current_City_Years  \\\n",
       "0  P00069042       0    1          10                          2   \n",
       "1  P00248942       0    1          10                          2   \n",
       "2  P00087842       0    1          10                          2   \n",
       "3  P00085442       0    1          10                          2   \n",
       "4  P00285442       1    7          16                          4   \n",
       "\n",
       "   Marital_Status  Product_Category_1  Product_Category_2  Product_Category_3  \\\n",
       "0               0                   3                 8.0                16.0   \n",
       "1               0                   1                 6.0                14.0   \n",
       "2               0                  12                 8.0                16.0   \n",
       "3               0                  12                14.0                16.0   \n",
       "4               0                   8                 8.0                16.0   \n",
       "\n",
       "   Purchase  B  C  \n",
       "0    8370.0  0  0  \n",
       "1   15200.0  0  0  \n",
       "2    1422.0  0  0  \n",
       "3    1057.0  0  0  \n",
       "4    7969.0  0  1  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "7f64a3da",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 783667 entries, 0 to 233598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   Product_ID                  783667 non-null  object \n",
      " 1   Gender                      783667 non-null  int64  \n",
      " 2   Age                         783667 non-null  int64  \n",
      " 3   Occupation                  783667 non-null  int64  \n",
      " 4   Stay_In_Current_City_Years  783667 non-null  object \n",
      " 5   Marital_Status              783667 non-null  int64  \n",
      " 6   Product_Category_1          783667 non-null  int64  \n",
      " 7   Product_Category_2          783667 non-null  float64\n",
      " 8   Product_Category_3          783667 non-null  float64\n",
      " 9   Purchase                    550068 non-null  float64\n",
      " 10  B                           783667 non-null  uint8  \n",
      " 11  C                           783667 non-null  uint8  \n",
      "dtypes: float64(3), int64(5), object(2), uint8(2)\n",
      "memory usage: 67.3+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "63b17936",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 783667 entries, 0 to 233598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   Product_ID                  783667 non-null  object \n",
      " 1   Gender                      783667 non-null  int64  \n",
      " 2   Age                         783667 non-null  int64  \n",
      " 3   Occupation                  783667 non-null  int64  \n",
      " 4   Stay_In_Current_City_Years  783667 non-null  int32  \n",
      " 5   Marital_Status              783667 non-null  int64  \n",
      " 6   Product_Category_1          783667 non-null  int64  \n",
      " 7   Product_Category_2          783667 non-null  float64\n",
      " 8   Product_Category_3          783667 non-null  float64\n",
      " 9   Purchase                    550068 non-null  float64\n",
      " 10  B                           783667 non-null  uint8  \n",
      " 11  C                           783667 non-null  uint8  \n",
      "dtypes: float64(3), int32(1), int64(5), object(1), uint8(2)\n",
      "memory usage: 64.3+ MB\n"
     ]
    }
   ],
   "source": [
    "##convert object into integers\n",
    "df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "b03cbf15",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['B']=df['B'].astype(int)\n",
    "df['C']=df['C'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "2abcf60e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 783667 entries, 0 to 233598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                      Non-Null Count   Dtype  \n",
      "---  ------                      --------------   -----  \n",
      " 0   Product_ID                  783667 non-null  object \n",
      " 1   Gender                      783667 non-null  int64  \n",
      " 2   Age                         783667 non-null  int64  \n",
      " 3   Occupation                  783667 non-null  int64  \n",
      " 4   Stay_In_Current_City_Years  783667 non-null  int32  \n",
      " 5   Marital_Status              783667 non-null  int64  \n",
      " 6   Product_Category_1          783667 non-null  int64  \n",
      " 7   Product_Category_2          783667 non-null  float64\n",
      " 8   Product_Category_3          783667 non-null  float64\n",
      " 9   Purchase                    550068 non-null  float64\n",
      " 10  B                           783667 non-null  int32  \n",
      " 11  C                           783667 non-null  int32  \n",
      "dtypes: float64(3), int32(3), int64(5), object(1)\n",
      "memory usage: 68.8+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "5001ec19",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Age', ylabel='Purchase'>"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Visualisation Age vs Purchased\n",
    "sns.barplot('Age','Purchase',hue='Gender',data=df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43cbe250",
   "metadata": {},
   "source": [
    "## Purchasing of men is high then women"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "30fcc329",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Occupation', ylabel='Purchase'>"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Visualization of Purchase with occupation\n",
    "sns.barplot('Occupation','Purchase',hue='Gender',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "769093d5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Product_Category_1', ylabel='Purchase'>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "cef905fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Product_Category_2', ylabel='Purchase'>"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "0e95ec78",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Product_Category_3', ylabel='Purchase'>"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "1423d576",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>8370.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1057.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>7969.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation  Stay_In_Current_City_Years  \\\n",
       "0  P00069042       0    1          10                           2   \n",
       "1  P00248942       0    1          10                           2   \n",
       "2  P00087842       0    1          10                           2   \n",
       "3  P00085442       0    1          10                           2   \n",
       "4  P00285442       1    7          16                           4   \n",
       "\n",
       "   Marital_Status  Product_Category_1  Product_Category_2  Product_Category_3  \\\n",
       "0               0                   3                 8.0                16.0   \n",
       "1               0                   1                 6.0                14.0   \n",
       "2               0                  12                 8.0                16.0   \n",
       "3               0                  12                14.0                16.0   \n",
       "4               0                   8                 8.0                16.0   \n",
       "\n",
       "   Purchase  B  C  \n",
       "0    8370.0  0  0  \n",
       "1   15200.0  0  0  \n",
       "2    1422.0  0  0  \n",
       "3    1057.0  0  0  \n",
       "4    7969.0  0  1  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "091ed800",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Feature Scaling \n",
    "df_test=df[df['Purchase'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "6c7b01e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train=df[~df['Purchase'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "bd4ef1e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=df_train.drop('Purchase',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "9228eef1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>8.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Product_ID  Gender  Age  Occupation  Stay_In_Current_City_Years  \\\n",
       "0  P00069042       0    1          10                           2   \n",
       "1  P00248942       0    1          10                           2   \n",
       "2  P00087842       0    1          10                           2   \n",
       "3  P00085442       0    1          10                           2   \n",
       "4  P00285442       1    7          16                           4   \n",
       "\n",
       "   Marital_Status  Product_Category_1  Product_Category_2  Product_Category_3  \\\n",
       "0               0                   3                 8.0                16.0   \n",
       "1               0                   1                 6.0                14.0   \n",
       "2               0                  12                 8.0                16.0   \n",
       "3               0                  12                14.0                16.0   \n",
       "4               0                   8                 8.0                16.0   \n",
       "\n",
       "   B  C  \n",
       "0  0  0  \n",
       "1  0  0  \n",
       "2  0  0  \n",
       "3  0  0  \n",
       "4  0  1  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "fd0d9abe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(550068, 11)"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "170b94c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=df_train['Purchase']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "56b1e6ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(550068,)"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "f062eb3a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0          8370.0\n",
       "1         15200.0\n",
       "2          1422.0\n",
       "3          1057.0\n",
       "4          7969.0\n",
       "           ...   \n",
       "550063      368.0\n",
       "550064      371.0\n",
       "550065      137.0\n",
       "550066      365.0\n",
       "550067      490.0\n",
       "Name: Purchase, Length: 550068, dtype: float64"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "f94362e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "     X, y, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "1ca71bde",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\win10\\anaconda3\\lib\\site-packages\\pandas\\core\\frame.py:4906: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  return super().drop(\n"
     ]
    }
   ],
   "source": [
    "X_train.drop('Product_ID',axis=1,inplace=True)\n",
    "X_test.drop('Product_ID',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "409b48fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "## feature Scaling \n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc=StandardScaler()\n",
    "X_train=sc.fit_transform(X_train)\n",
    "X_test=sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fa7b4df",
   "metadata": {},
   "outputs": [],
   "source": [
    "## train ur model"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
