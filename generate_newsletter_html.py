import torch
import joblib
from transformers import (
	AutoTokenizer, 
    AutoModelForSequenceClassification,
)
import pandas as pd
import re
import datetime
import html


# Configuration

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

NEW_DATA_INPUT_FILE = "Data/TECH_NL_CRAWL_FINAL_FULL_DATA_2026-05-27.xlsx"
MODEL_PATH = "Models/newsletter_transformer_v1"
OUTPUT_FILE = f"Data/New_Articles_Predicted_{timestamp}.xlsx"
NL_NUMBER = 19
GENERATED_HTML_FILE = f"yoon-newsletter-email-nr-{NL_NUMBER}.html"


# Load models

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load(f"Models/newsletter_label_encoder_v1.pkl")


# Use CPU by default and change to cuda if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

print("Model loaded")
print("Device:", device)


# Define predict function

def predict_text(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    label = label_encoder.inverse_transform([pred_idx])[0]

    return label, confidence


# Load dataset 

df = pd.read_excel(NEW_DATA_INPUT_FILE)


# Remove duplicates

df.drop_duplicates(subset='article_url', inplace=True)


# Join columns for prediction

df['Content'] = (df['title'].fillna('') + ' ' + df['tags'].fillna('')).str.strip()
print(df['Content'])


# Remove rows with empty cells

df = df[df['Content'].notna()].copy()


## Transform Content column to lowercase

df['Content'] = df['Content'].fillna('').str.lower()
print(df['Content'])


## Remove lines with Japanese/Chinese/Korean text

def contains_cjk(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

df = df[~df['Content'].apply(contains_cjk)]


# Loop over dataset and classify content

predictions = []
confidences = []

for i, row in df.iterrows():
    try:
        text = row['Content']
        label, conf = predict_text(text)
        predictions.append(label)
        confidences.append(round(conf * 100, 2))
        print(f"{i}: {label} ({conf*100:.2f}%)")
    except Exception as e:
        predictions.append('ERROR')
        confidences.append('ERROR')
        print(f'An error occurred during classification: {e}')
        

# Add predicted data to dataframe

df["Predicted_Subcategory"] = predictions 
df["Confidence"] = confidences


# Sort rows based on categories

df = df.sort_values(by="Predicted_Subcategory")
 
df.to_excel(OUTPUT_FILE, index=False)

print("Finished.")
print("Saved:", OUTPUT_FILE)


# Generate html newsletter

html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yoon Tech Briefing</title>
    <style>
        body {{
            font-family: 'Roboto', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}

        .newsletter {{
            max-width: 600px;
            margin: 10px auto;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}

		.rounded-circle {{
			border-radius:50%!important;
			width: 55px;
		}}

		.robot-image {{
			margin-top: 30px;
			margin-bottom: 15px;
			width: 40px;
			display: block;
			margin-left: auto;
			margin-right: auto;
		}}

		.social-media {{
			font-size: small;
			color: #dddddd
		}}

        .header {{
            background-color: #412066;
            color: white;
            padding: 10px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}

		hr {{
			height: 0px;
			border: none;
			border-top: 1px solid rgb(206, 206, 206);
		}}

        .content {{
            padding: 20px;
			background-color: #f0f0f0;
        }}

        .content h2 {{
            font-size: 20px;
            color: #333333;
            margin-bottom: 10px;
        }}

        .content p {{
			font-size: 11pt;
            line-height: 1.5;
            color: #353535;
        }}

		.content li {{
			line-height: 1.5;
			font-size: 11pt;
			color: #353535;
		}}

		.content a {{
			color: inherit;
			text-decoration: none;
		}}

		.content a:hover {{
			text-decoration: underline;
		}}


		.news-section p {{
			color: #585858;
		}}

		.social-media a {{
			color: inherit;
			text-decoration: none;
		}}

		.social-media a:hover {{
			text-decoration: underline;
		}}

		.social-media-items {{
			margin-left: 5px;
			margin-right: 5px;
		}}

        .cta {{
            display: block;
            text-align: center;
            margin: 20px 0;
        }}

        .cta a {{
            text-decoration: none;
            background-color: #330062;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 14px;
            transition: background-color 0.3s;
        }}

        .cta a:hover {{
            background-color: #330062;
        }}

		.section-divider {{
			margin-top: 45px;
			width: 60%;
		}}

		.links-category {{
			font-size: 12pt;
			text-align: center;
			margin-top: 45px;
			margin-bottom: 20px;
			color: #444444;
		}}

		.link-source {{
			font-style: normal;
			font-size: small;
			margin-right: 5px;
		}}

		.links-text {{
			text-align: center;
			font-style: italic;
			padding-left: 20px;
			padding-right: 20px;
		}}

		.links-icon {{
			width: 16px;
			margin-right: 10px;
		}}

		.lightning-image {{
			margin-top: 15px;
			margin-bottom: 15px;
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 200px;
			font-size: small;
		}}

        .footer {{
            background-color: #333;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #888888;
        }}
		
		.footer p {{
			color: #fff;
		}}

        .footer a {{
            color: #d1d1d1;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
	<p style="font-size: small; color: #3a3a3a; text-align: center;">
		If you're having trouble viewing this email, <a style="text-decoration: none; color: #3a3a3a;" 
		href="https://www.yoon-dev.com/media/media/resources/yoon-newsletter-server-nr-{NL_NUMBER}.html" target="_blank">
		<span style="text-decoration: underline">click here to open it in your browser</span></a>
	</p>
    
	<div class="newsletter">
        <div class="header">
        
            <p class="social-media">
                <span class="social-media-items"><a href="https://www.yoon-dev.com/" target="_blank">Website</a></span> | 
                <span class="social-media-items"><a href="https://twitter.com/cyoon_dev" target="_blank">Twitter</a></span> | 
                <span class="social-media-items"><a href="https://www.instagram.com/yoon_dev" target="_blank">Instagram</a></span> | 
                <span class="social-media-items"><a href="https://github.com/ChristopherYoon89" target="_blank">GitHub</a></span>  
            </p>
            
            <h3>CHRIS YOON DEVELOPING</h3>
            
            <img class="rounded-circle" style="font-size:small;" src="https://www.yoon-dev.com/static/blog/media/profile-pic.webp" alt="Profile Picture">
            
            <h4 style="font-size: small; ">Tech Briefing #{NL_NUMBER}</h4>
			
        </div>
        
		<div class="content">			
		
        <h2 style="margin-top: 15px;">Header of article</h2>

        <p>
        Paragraph 1
        </p>

        <p>
        For more information feel free to <a style="text-decoration: underline;" href="https://www.yoon-dev.com/contact/" target="_blank">reach 
        out</a> to us and check out our social media channels - I would be excited to get in touch with you!
        </p>

        <p>
        <img class="robot-image" src="https://www.yoon-dev.com/static/blog/media/robotfreeze.png" alt="robot image">
        </p>

        <div class="links-text" style="margin-top: 15px;">
        <p style="color: #888888;">
        The articles shared in the newsletter do not necessarily reflect my personal views.
        </p>
        </div>

        <div class="news-section">
"""


# Order categories

category_order = [
    "Software & Apps",
    "Hardware",
    "Artificial Intelligence",
    "Security & Hacking",
    "Surveillance & Censorship",
    "Bitcoin",
    "Gaming",
    "Science",
    "Engineering",
    "Other Good Reads"
]

df["Predicted_Subcategory"] = pd.Categorical(
    df["Predicted_Subcategory"],
    categories=category_order,
    ordered=True
)


# Populate news articles by category

for category, group in df.groupby("Predicted_Subcategory", sort=True):

    html_output += """
    <hr class="section-divider">
    
    """

    html_output += f'<h4 class="links-category">{category}</h4>'
    html_output += '<div class="links-text">'

    # Loop articles
    for _, row in group.iterrows():
        title = html.escape(str(row.get("title", "No title")))
        print(title)
        url = html.escape(str(row.get("article_url", "#")))
        print(url)
        source = html.escape(str(row.get("source_name", "Unknown")))
        (source)

        html_output += f"""
        <p>
            <span class="link-source">[{source}]</span>
            <a href="{url}" target="_blank">
                {title}
            </a>
        </p>
        """

    html_output += "</div>"


# Add CTA and footer to html

html_output += """

			<div class="links-text">
			
				<div class="cta"  style="margin-bottom: 65px; margin-top: 65px; font-style: normal;">
					<a href="https://www.yoon-dev.com/newsletter" target="_blank">Sign up for newsletter</a>
				</div>

				<hr class="section-divider">

				<h4 class="links-category">
				Enjoying this newsletter? If it's adding value to your day, you can donate a 
				few sats or buy me a coffee! Every little bit helps to keep this going. Thank you for your support!
				</h4>

                <p style="margin-top: 35px;">
				Buy me a coffee
				</p>
                <a href="https://www.yoon-dev.com/media/media/resources/buy-me-a-coffee-qr-code.png" target="_blank">
				<img class="lightning-image" src="https://www.yoon-dev.com/media/media/resources/buy-me-a-coffee-qr-code.png" alt="Buy me a coffee QR Code">
				</a>

				<p style="margin-top: 35px;">
				Bitcoin Address:
				</p>
				<p class="lightning-address">
				13fm7Vxz5qZV55h789uCRGxQegw5LGCBRn
				</p>
				<a href="https://www.yoon-dev.com/media/media/resources/BitcoinQRCode.png" target="_blank">
				<img class="lightning-image" src="https://www.yoon-dev.com/media/media/resources/BitcoinQRCode.png" alt="Bitcoin Wallet QR Code">
                </a>
				
				<p style="margin-top: 35px;">
				Bitcoin Lightning Address:
				</p>
				<p class="lightning-address">
				<span style="word-wrap: break-word; overflow-wrap: break-word;">
                specialrescue097@walletofsatoshi.com
                </span>
				</p>
                <a href="https://www.yoon-dev.com/media/media/resources/WalletOfSatoshi.jpg" target="_blank">
				<img class="lightning-image" src="https://www.yoon-dev.com/media/media/resources/Bitcoin_lightning_qrcode.png" alt="Lightning Wallet QR Code">
				</a>
				
				<hr class="section-divider">
				<div style="margin-bottom: 15px; margin-top: 25px; font-style: normal;">
					<p class="social-media">
						<span class="social-media-items"><a href="https://www.yoon-dev.com/" target="_blank">Website</a></span> | 
						<span class="social-media-items"><a href="https://twitter.com/cyoon_dev" target="_blank">Twitter</a></span> | 
						<span class="social-media-items"><a href="https://www.instagram.com/yoon_dev" target="_blank">Instagram</a></span> | 
						<span class="social-media-items"><a href="https://github.com/ChristopherYoon89" target="_blank">GitHub</a></span> 
					</p>
				</div>
			</div>
		</div>
	</div>
		
<div class="footer">
	<p>You are receiving this email because you subscribed to our newsletter.</p>
	<p>
	<a href="https://www.yoon-dev.com/newsletter-unsubscribe/" target="_blank">Unsubscribe</a> | 
	<a href="https://www.yoon-dev.com/contact/" target="_blank">Contact</a>
	</p>
</div>
    
</body>
</html>

"""


with open(GENERATED_HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html_output)