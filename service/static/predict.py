import os


import static.train_model as train

# A function to do it
def gender_predictor_mnb(a, cv, clf):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        return "Female"
    else:
        return "Male"


def gender_predictor_dt(a, dv, dclf):
# Build Features and Transform them
    test_name = [a]
    transform_dv =dv.transform(train.features(test_name))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        return "Female"
    else:
        return "Male"
