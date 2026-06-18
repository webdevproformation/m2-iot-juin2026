vous avez les données suivantes :

data set => 

student_data1
  student_id              name  marks
0         S1  Danniella Fenton    200
1         S2      Ryder Storey    210
2         S3      Bryce Jensen    190
3         S4         Ed Bernal    222
4         S5       Kwame Morin    199

```py
student_data1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'], 
        'marks': [200, 210, 190, 222, 199]})
```

Écrivez un programme Pandas pour ajouter la ligne suivante à un DataFrame existant et afficher les données combinées.

nouvelle ligne à ajouter 

New Row(s)
student_id                  S6
name          Scarlette Fisher
marks                      205