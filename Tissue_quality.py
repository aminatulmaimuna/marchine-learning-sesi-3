from sklearn.neighbors import KNeighborsClassifier

# Data training
#training data
X_train = [[8, 4], [4,5], [4,6],[7,7],[5,6],[6,5]]
y_train = ['Baik', 'Jelek', 'Jelek','Baik','Jelek','Baik']

# Inisialisasi model KNN dengan jumlah tetangga (n_neighbors) sebesar 3 (ubah nilai k)
## Initialize the KNN model with the number of neighbors (n_neighbors) of 3 (change the value of k)
knn = KNeighborsClassifier(n_neighbors=4)

# Melatih model dengan data training
#train model using data training
knn.fit(X_train, y_train)

# Data test
X_test = [[2,5]]
y_test = ['Baik']  # Label yang seharusnya didapat dari prediksi

# Melakukan prediksi dengan model yang sudah dilatih
predictions = knn.predict(X_test)

# Mengevaluasi performa model
accuracy = knn.score(X_test, y_test)
print("Akurasi:", accuracy)

# Menampilkan hasil prediksi
print("Hasil Prediksi:")
for i, pred in enumerate(predictions):
    print(f"Data test {i+1}: Kelas prediksi = {pred}")

