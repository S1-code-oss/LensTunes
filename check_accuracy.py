import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

model = load_model('emotion_model.keras')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory('test', target_size=(48,48),
                                          batch_size=64, color_mode='grayscale',
                                          shuffle=False)

loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

print("\nPer-emotion breakdown:")
print(classification_report(y_true, y_pred, target_names=emotions))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nSaved confusion_matrix.png")