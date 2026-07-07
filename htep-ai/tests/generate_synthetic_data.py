import os
import sys
import json
import random
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_dict(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

drugs = load_dict(os.path.join(project_root, "data", "dictionaries", "drugs.txt"))
diseases = load_dict(os.path.join(project_root, "data", "dictionaries", "diseases.txt"))

if not drugs: drugs = ["Metformin", "Aspirin", "Amoxicillin", "Lisinopril", "Atorvastatin"]
if not diseases: diseases = ["Diabetes", "Hypertension", "Asthma", "Anemia"]

first_names = ["John", "Jane", "Robert", "Emily", "Michael", "Sarah", "David", "Laura", "James", "Emma"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
hospitals = ["CityCare Hospital", "General Hospital", "Mercy Medical Center", "Sunrise Clinic"]
doctors = ["Dr. Smith", "Dr. Adams", "Dr. Kapoor", "Dr. Lee"]
frequencies = ["once daily", "twice daily", "thrice daily", "every 8 hours", "before sleep"]

def generate_patient_data():
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = str(random.randint(20, 80))
    gender = random.choice(["Male", "Female"])
    date = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2025"
    doctor = random.choice(doctors)
    hospital = random.choice(hospitals)
    
    num_diseases = random.randint(1, 2)
    pat_diseases = random.sample(diseases, num_diseases)
    
    num_meds = random.randint(1, 3)
    pat_meds = []
    for _ in range(num_meds):
        pat_meds.append({
            "name": random.choice(drugs).title(),
            "dosage": f"{random.choice([5, 10, 50, 100, 250, 500])}mg",
            "frequency": random.choice(frequencies)
        })
        
    instructions_bank = [
        "- Return to clinic if symptoms worsen.",
        "- Continue medications as prescribed.",
        "- Avoid high-sugar foods.",
        "- Drink plenty of fluids.",
        "- Schedule a follow-up in 2 weeks."
    ]
    instructions = random.sample(instructions_bank, 2)
    instructions.append("Signature:")
    
    lines = []
    lines.append(f"{hospital} - Discharge Summary")
    lines.append(f"Patient Name : {name}")
    lines.append(f"Patient ID : CH-{random.randint(1000, 9999)}")
    lines.append(f"Age : {age}")
    lines.append(f"Gender : {gender}")
    lines.append(f"Date : {date}")
    lines.append("Diagnosis:")
    for d in pat_diseases:
        lines.append(f"- {d}")
    lines.append("Treatment Summary:")
    for m in pat_meds:
        lines.append(f"- {m['name']} {m['dosage']} {m['frequency']}")
    lines.append("Follow up instructions:")
    for inst in instructions:
        lines.append(inst)
    lines.append(f"Consultant: {doctor}")
    
    raw_text = "\n".join(lines)
    
    extracted = {
        "patient_name": name,
        "age": age,
        "gender": gender,
        "date": date,
        "doctor_name": doctor,
        "hospital": hospital,
        "diagnosis": pat_diseases,
        "medicines": pat_meds,
        "vitals": {},
        "instructions": instructions + [f"Consultant: {doctor}"]
    }
    
    return raw_text, extracted

def render_image(text, filepath, font_type):
    try:
        if font_type == "handwritten":
            font = ImageFont.truetype("segoepr.ttf", 24)
        else:
            font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    lines = text.split('\n')
    line_height = 25
    img_width = 600
    img_height = max(400, len(lines) * line_height + 40)
    
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)
    
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += line_height
        
    image.save(filepath)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic medical documents.")
    parser.add_argument("--count", type=int, default=3, help="Number of images to generate")
    parser.add_argument("--wipe", action="store_true", help="Wipe existing ground_truth.json")
    args = parser.parse_args()
    
    gt_file = os.path.join(project_root, "tests", "ground_truth.json")
    raw_dir = os.path.join(project_root, "data", "raw")
    
    if args.wipe:
        existing_data = []
    else:
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            existing_data = []
            
    start_id = 1
    for item in existing_data:
        if "synthetic_" in item["image_path"]:
            try:
                num = int(item["image_path"].split("synthetic_")[1].split(".png")[0])
                if num >= start_id:
                    start_id = num + 1
            except:
                pass

    print(f"Generating {args.count} synthetic images...")
    
    for i in range(args.count):
        img_id = start_id + i
        filename = f"synthetic_{img_id}.png"
        filepath = os.path.join(raw_dir, filename)
        
        raw_text, extracted = generate_patient_data()
        
        # Randomly choose between printed and handwritten
        font_type = random.choice(["printed", "handwritten"])
        render_image(raw_text, filepath, font_type)
        
        existing_data.append({
            "image_path": f"data/raw/{filename}",
            "type": font_type,
            "raw_text": raw_text,
            "extracted_data": extracted
        })
        print(f"Generated {filepath}")
        
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)
        
    print(f"\nSuccessfully generated {args.count} images and updated ground_truth.json!")

if __name__ == "__main__":
    main()
