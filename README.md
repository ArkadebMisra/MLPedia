# MLPedia

MLpedia is supposed to be a ML playground. the web application can apply various ML algorithms to simple data and saved the learned models to database. users can later use this database. There are also data visualization tools available too.

# How to use
I have to say even though I am quite happy with the project, it is very counter intutive and it does not have lot of guidance in how to use it. hence this section. 

### Plotter

*  The **plotter** section has some very simple plotting tools. which are farely self explanatory. you can understand very easily what to do by looking at the forms. The only important thing to here is how to enter the **X values** and **Y values** . Enter same number of data points in both the field and make sure they are **comma seperated** with a **single space after each comma**. as an example consider the following,  also added a screenshot- </br>

**Line1 X values: 5, 10, 15, 20**</br>
**Line1 Y values: 17, 23, 5, 10**</br>

![image28](https://github.com/user-attachments/assets/17abfcf6-a6f0-4850-adb1-ecde84b95797)


### Analytica
* To use the **Analytica** section first go ahed and make an account. this step is necessary if you want to use the analytica section of the application

* Most of the algo here takes in **two files**. One of which is the **features file**. this is a **simple csv** file. each line describes the feature input of a single data point. the other one is **labels file**. This is also a csv file but containing a single value each line. **the value of ith line** of this file should be the **lable of ith input feature of the features file** . There is one more **important thing to note. in case of neural network the first line of the labels file is not the conventional field name of a csv. it should be modified to the number of labels that are present in the file**. I have provided some example files to try with. here is the link [example inputs](https://drive.google.com/drive/folders/11QFZB1bkxJo5vE_g_lu5FiYEMqUF0T_S?usp=sharing) 

* after the learned model is saved you can do prediction on new datapoint.(alas! you can do it on anly one datapoint at a time, future scope of work.). **again fill in a new feature vactor in the form and hit submit. features vector is values seperated by commas. again remember to add a single space after each value.**

* Remember this site is supposed to handle simple datasets. algorythms does not have normalization layers and probably will suffer from vanishing gradiant problem.(future scope of work)

* a screenshot </br>
![image30](https://github.com/user-attachments/assets/c61dd1d7-1310-48a0-8570-e6e4da91fe2e)

# Screenshots

![1660361277740](https://github.com/user-attachments/assets/26de4821-0fad-4f4e-a4ba-0221a11a918b)
![1660361282179](https://github.com/user-attachments/assets/24f1db4d-7f95-4e5a-a7dc-c05bb654018e)
![1660361280442](https://github.com/user-attachments/assets/af44d793-603b-4dee-a801-b049ceadfc49)
![1660361280455](https://github.com/user-attachments/assets/030aa4b2-1acf-47fa-9447-42c1b7f19420)
![1660361281144](https://github.com/user-attachments/assets/c06eaab7-318c-475b-8bfa-c4081818af7c)
![1660361279380](https://github.com/user-attachments/assets/d1cf61a7-6d4d-4cb0-815f-48ee57ac2f45)


