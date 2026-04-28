from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\n📊 Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save report
    with open("output/report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nAccuracy: {accuracy}")