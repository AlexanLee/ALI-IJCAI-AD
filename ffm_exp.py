import ffm

# prepare the data
# (field, index, value) format

X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
     [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
     [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],
     [(1, 0, 1), (2, 3, 1), (3, 5, 1)], ]

y = [1, 1, 0, 1]

ffm_data = ffm.FFMData(X, y)
ffm_data_test = ffm.FFMData(X, y)

model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
model.fit(ffm_data, num_iter=10, val_data=ffm_data_test, metric='auc', early_stopping=6,
          maximum=True)

print(model.predict_proba(ffm_data_test))

model.save_model('result/ololo.bin')

model = ffm.read_model('result/ololo.bin')

print(model.predict(ffm_data_test))
