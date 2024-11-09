from flask import Flask, request, render_template, jsonify, send_from_directory
import fitz  # PyMuPDF for PDF processing
import os
from transformers import  CLIPProcessor, CLIPModel
from PIL import Image
import PyPDF2

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize models

# Image classification using CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels for classification


image_labels = ["passport page", "front and back page of passport","front page of passport","last two page of a passport","passport size photo","ID Photo (Single Photograph)","ID card","driving license","exam marksheet","certificate","document page", "text-only page","visa","letter"]


# Function to classify images with CLIP
def classify_image(image_path,threshold=0.3):
    image = Image.open(image_path)
    inputs = clip_processor(text=image_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    
    # Get the predicted label and confidence score
    max_prob, max_idx = probs.max(dim=1)
    predicted_label = image_labels[max_idx]
    
    # if predicted_label != "bangladesh passport":
    if max_prob.item() < threshold:
            predicted_label = "not a passport page"
    
    return predicted_label, max_prob.item()



# Extract images from PDF and classify using text and image models
def classify_pages(pdf_path):
    pdf = fitz.open(pdf_path)
    passport_pages, other_pages = [], []

    for page_num in range(pdf.page_count):
        page = pdf[page_num]

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"page_{page_num}.png")
        page.get_pixmap().save(img_path)
        # Image-based classification
        img_label,confidence = classify_image(img_path)
                
        if img_label in ["passport page", "ID Photo (Single Photograph)","front and back page of passport","front page of passport","last two page of a passport","passport size photo"]:
            passport_pages.append((pdf_path, page_num,img_label,confidence))
        else:
            other_pages.append((pdf_path, page_num, img_label,confidence))
            os.remove(img_path)  # Clean up
 

    pdf.close()
    return passport_pages, other_pages

# Save classified pages to output PDFs
def save_pages_to_pdf(page_list, output_filename):  
    pdf_writer = PyPDF2.PdfWriter()
    for pdf_file, page_num,label,confidence in page_list:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pdf_writer.add_page(pdf_reader.pages[page_num])
    with open(output_filename, "wb") as out_pdf:
        pdf_writer.write(out_pdf)

# Flask route for file upload and processing
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        pdf_files = []
        classifications=[]

        for file in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            pdf_files.append(file_path)

        # Classify pages in PDFs
        passport_pages, other_pages = [], []
        for pdf_path in pdf_files:
            pp, op = classify_pages(pdf_path)
            passport_pages.extend(pp)
            for i in pp :
                classifications.append({"pdf":"Passport","page":"","label":i[2],"confidence":i[3]})
            other_pages.extend(op)
            for i in op :
                classifications.append({"pdf":"other","page":"","label":i[2],"confidence":i[3]})
            

        # Save classified pages to output PDFs
        passport_output = os.path.join(app.config['OUTPUT_FOLDER'], "Passport_Photo_Pages.pdf")
        other_output = os.path.join(app.config['OUTPUT_FOLDER'], "Other_Pages.pdf")
        save_pages_to_pdf(passport_pages, passport_output)
        save_pages_to_pdf(other_pages, other_output)

        return render_template("index.html",classifications=str(classifications), passport_file="Passport_Photo_Pages.pdf", other_file="Other_Pages.pdf")

    return render_template("index.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)