import os
import torch
from flask import Flask, abort, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField

from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder

from utils.utils import adaptive_instance_normalization

# **********APP CONFIG**********

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["CONTENT_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "content")
app.config["STYLE_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "style")
app.config["OUTPUT_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "outputs")

app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CONTENT_FOLDER"], exist_ok=True)
os.makedirs(app.config["STYLE_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


# **********FORM**********


class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField("Style Strength", default=1.0)

    submit = SubmitField("Transfer Style")


# **********DEVICE**********

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# **********LOAD MODELS**********

encoder = VGGEncoder("vgg_normalised.pth").to(device)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load("experiments/final_exp/decoder_final.pth", map_location=device)
)

encoder.eval()
decoder.eval()

print("Models Loaded")


# **********HELPERS**********


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(content_img, style_img, alpha=1.0):

    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )

    content = transform(content_img).unsqueeze(0).to(device)

    style = transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():

        content_features = encoder(content, is_test=True)

        style_features = encoder(style, is_test=True)

        stylized_features = adaptive_instance_normalization(
            content_features, style_features
        )

        stylized_features = alpha * stylized_features + (1 - alpha) * content_features

        output = decoder(stylized_features)

    return output


def save_tensor_image(tensor, path):

    tensor = tensor.cpu()

    tensor = tensor.squeeze(0)

    tensor = tensor.clamp(0, 1)

    image = transforms.ToPILImage()(tensor)

    image.save(path)


# **********ROUTES**********


@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None

    content_image = None

    style_image = None

    error = None

    if form.validate_on_submit():

        try:

            # CONTENT IMAGE

            if form.content.data and form.content.data.filename:

                content_name = "content_" + secure_filename(form.content.data.filename)

                content_path = os.path.join(app.config["CONTENT_FOLDER"], content_name)

                form.content.data.save(content_path)

                form.content_path.data = content_name

            else:

                content_name = form.content_path.data

            # STYLE IMAGE

            if form.style.data and form.style.data.filename:

                style_name = "style_" + secure_filename(form.style.data.filename)

                style_path = os.path.join(app.config["STYLE_FOLDER"], style_name)

                form.style.data.save(style_path)

                form.style_path.data = style_name

            else:

                style_name = form.style_path.data

            if content_name and style_name:

                content_image = content_name

                style_image = style_name

                content_path = os.path.join(app.config["CONTENT_FOLDER"], content_name)

                style_path = os.path.join(app.config["STYLE_FOLDER"], style_name)

                content = Image.open(content_path).convert("RGB")

                style = Image.open(style_path).convert("RGB")

                alpha = float(form.alpha.data)

                alpha = max(0, min(alpha, 1))

                output = style_transfer(content, style, alpha)

                result_name = "stylized_" + content_name

                result_path = os.path.join(app.config["OUTPUT_FOLDER"], result_name)

                save_tensor_image(output, result_path)

                result_image = result_name

        except Exception as e:

            error = str(e)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_image,
        style_image=style_image,
        error=error,
    )


@app.route("/uploads/<filename>")
def send_image(filename):

    search_folders = [
        app.config["OUTPUT_FOLDER"],
        app.config["CONTENT_FOLDER"],
        app.config["STYLE_FOLDER"],
    ]

    for folder in search_folders:

        if os.path.exists(os.path.join(folder, filename)):

            return send_from_directory(folder, filename)

    abort(404)


@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory("static/assets/examples", filename)


# **********RUN**********

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
