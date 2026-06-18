# vous avez les dictionnaires / données suivantes :

vous avez les données suivantes :

student_data1:
  student_id              name  marks
0         S1  Danniella Fenton    200
1         S2      Ryder Storey    210
2         S3      Bryce Jensen    190
3         S4         Ed Bernal    222
4         S5       Kwame Morin    199

student_data2:
  student_id              name  marks
0         S4  Scarlette Fisher    201
1         S5  Carla Williamson    200
2         S6       Dante Morse    198
3         S7    Kaiser William    219
4         S8   Madeeha Preston    201

exam_data:
   student_id  exam_id
0          S1       23
1          S2       45
2          S3       12
3          S4       67
4          S5       21
5          S7       55
6          S8       33
7          S9       14
8         S10       56
9         S11       83
10        S12       88
11        S13       12

```py
student_data1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'], 
        'marks': [200, 210, 190, 222, 199]})

student_data2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'], 
        'marks': [201, 200, 198, 219, 201]})

exam_data = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
        'exam_id': [23, 45, 12, 67, 21, 55, 33, 14, 56, 83, 88, 12]})
```


La Question :

Écrivez un programme Pandas pour joindre les deux dataframes donnés le long des lignes et les fusionner avec un autre dataframe le long de la colonne commune id.