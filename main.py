import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def classify_image(image):
    inputs = processor(text=["inside the store", "store appearance", "food", "etc"], images=image, return_tensors="pt",
                       padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return {
        "가게 내부": probs[0][0].item(),
        "가게 외부": probs[0][1].item(),
        "음식": probs[0][2].item(),
        "기타": probs[0][3].item()
    }


# Streamlit web app
def main():
    # Set Streamlit app title
    st.title("이미지 분류")

    # File uploader
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = Image.open(image_file)
        # st.image(image, caption='Uploaded Image', use_column_width=True, width=20)

        # Run image classification
        category = classify_image(image)
        sorted_category = sorted(category.items(), key=lambda x: x[1], reverse=True)[0][0]
        st.write(f'이 사진은 {sorted_category} 사진입니다.')
        st.json(category)
        # st.write(f"Category: {category}")


# Run the Streamlit app
if __name__ == '__main__':
    main()
