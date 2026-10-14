# src/nlp/medical_extractor.py
"""
Structured Medical Document Extraction (JSON Output)
Extracts patient info, diagnosis, medicines, vitals, etc. using three layers:
1. Regex Patterns (Always available)
2. SpaCy NER (PERSON, ORG, DATE)
3. SciSpaCy NER (CHEMICAL, DISEASE) + Dictionary Matching
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

# Attempt to load spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not installed — falling back to regex/dictionary only")

DOSAGE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?|tablets?|capsules?|drops?))', re.IGNORECASE
)

# Longer phrases come first so "morning and night" wins over "morning",
# and "twice daily" over "daily".
FREQUENCY_PATTERN = re.compile(
    r'\b((?:once|twice|thrice)\s+(?:daily|a\s+day|a\s+week)'
    r'|\d+\s+times\s+(?:a\s+day|daily)'
    r'|morning\s+and\s+(?:night|evening)'
    r'|(?:before|after|with)\s+(?:meals?|food|breakfast|lunch|dinner|sleep)'
    r'|at\s+(?:bedtime|night)'
    r'|every\s+\d+\s+hours?'
    r'|[01]\s*-\s*[01]\s*-\s*[01]'
    r'|daily|bid|tid|qid|qds|tds|bd|od|hs|sos|prn|stat|morning|night)\b',
    re.IGNORECASE,
)

# Prescription line prefixes: "Tab. Paracetamol 500mg", "Cap Omez 20 mg"
DOSAGE_FORM_PREFIX = re.compile(r'^(?:tabs?|caps?|syp|syr|inj|oint|susp)\.?\s+', re.IGNORECASE)

# Degrees that trail a doctor's name: "Dr. Rao MBBS, MD"
DOCTOR_DEGREES = re.compile(r'(?:[\s,]+(?:MBBS|MD|MS|DNB|DM|MCh|FRCS|MRCP|BDS|MDS|DGO|DCH))+\.?$')


class MedicalDocExtractor:
    def __init__(self, drug_list: List[str], disease_list: List[str]):
        """
        Initializes the extractor with lists of drugs and diseases.
        Attempts to load Spacy models if available.
        """
        self.drugs = set(d.lower() for d in drug_list)
        self.diseases = set(d.lower() for d in disease_list)

        self.nlp = None
        self.sci_nlp = None

        if SPACY_AVAILABLE:
            # Load general NER model
            for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_trf"]:
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info("Loaded SpaCy model: %s", model_name)
                    break
                except Exception:
                    pass

            if not self.nlp:
                logger.warning("Could not load any SpaCy model (en_core_web_sm/md)")

            # Load SciSpacy NER model
            for model_name in ["en_ner_bc5cdr_md", "en_core_sci_sm"]:
                try:
                    self.sci_nlp = spacy.load(model_name)
                    logger.info("Loaded SciSpaCy model: %s", model_name)
                    break
                except Exception:
                    pass
            
            if not self.sci_nlp:
                logger.warning("Could not load any SciSpaCy model (en_ner_bc5cdr_md/en_core_sci_sm)")

    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extracts fields using regular expressions."""
        data = {
            "patient_name": None,
            "age": None,
            "gender": None,
            "date": None,
            "doctor_name": None,
            "hospital": None,
            "diagnosis": [],
            "medicines": [],
            "vitals": {},
            "instructions": []
        }

        # Date: e.g., 2026-07-01, 07/01/2026, 1st July 2026
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}(st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b', text, re.IGNORECASE)
        if date_match:
            data["date"] = date_match.group(1)

        # Patient Name: e.g., Patient: John Doe, Name: Jane Smith
        # Allows initials/apostrophes ("Mr. K. Suresh", "O'Brien") and drops the honorific
        name_match = re.search(
            r"(?:Patient(?:'s)?\s*Name|Patient|Name)\s*[:\-]\s*(?:(?:Mr|Mrs|Ms|Miss|Master|Baby)\.?\s+)?"
            r"([A-Za-z][A-Za-z.'\- \t]*?)(?=[ \t]*(?:\n|,|$)|\s+(?:Age|DOB|Date|Gender|Sex|Diagnosis|Rx|BP|ID|UHID|Mobile)\b)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if name_match:
            data["patient_name"] = name_match.group(1).strip()

        # Doctor Name: e.g., Dr. Smith, Doctor: Jane Doe
        # "Dr" must start a word and be followed by a capitalised name, so
        # "Drink plenty of fluids" and "syndrome" are not read as doctors.
        doc_match = re.search(
            r"(?<![A-Za-z])(?:(?i:Doctor|Physician|Consultant)\s*[:\-]\s*(?:(?i:Dr)\.?\s*)?|(?i:Dr)\.?\s*)"
            r"([A-Z][A-Za-z'\-]*\.?(?:[ \t]+[A-Z][A-Za-z'\-]*\.?){0,3})",
            text,
        )
        if doc_match:
            data["doctor_name"] = "Dr. " + DOCTOR_DEGREES.sub("", doc_match.group(1).strip())

        # Hospital/Clinic (Require a colon to avoid matching things like "Hospital - Summary")
        hosp_match = re.search(r'(?:Hospital|Clinic|Medical Center|Health Center)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if hosp_match:
            data["hospital"] = hosp_match.group(1).strip()
        else:
            # Letterhead line: "CityCare Hospital - Discharge Summary", "SUNRISE CLINIC"
            hosp_line = re.search(
                r"^[ \t]*((?:[A-Z][\w'&.]*[ \t]+){1,5}"
                r"(?i:Hospitals?|Clinic|Medical\s+Cent(?:er|re)|Health\s+Cent(?:er|re)|Nursing\s+Home|Polyclinic|Healthcare))\b",
                text,
                re.MULTILINE,
            )
            if hosp_line:
                data["hospital"] = hosp_line.group(1).strip()

        # Age/Gender
        age_match = re.search(r'(?:Age)\s*[:\-]\s*(\d+)', text, re.IGNORECASE)
        if age_match:
            data["age"] = age_match.group(1)
            
        gender_match = re.search(r'(?:Gender|Sex)\s*[:\-]\s*(Male|Female|M|F)', text, re.IGNORECASE)
        if gender_match:
            val = gender_match.group(1).upper()
            data["gender"] = "Male" if val in ("MALE", "M") else ("Female" if val in ("FEMALE", "F") else val)

        # Vitals
        bp_match = re.search(r'(?:BP|Blood Pressure)\s*[:\-]\s*(\d{2,3}\s*/\s*\d{2,3})', text, re.IGNORECASE)
        if bp_match:
            data["vitals"]["bp"] = bp_match.group(1).replace(" ", "")
            
        temp_match = re.search(r'(?:Temp|Temperature)\s*[:\-]\s*(\d{2,3}(?:\.\d)?\s*[FC])', text, re.IGNORECASE)
        if temp_match:
            data["vitals"]["temp"] = temp_match.group(1).replace(" ", "")
            
        pulse_match = re.search(r'(?:Pulse|HR|Heart Rate)\s*[:\-]\s*(\d{2,3})\s*(?:bpm)?', text, re.IGNORECASE)
        if pulse_match:
            data["vitals"]["pulse"] = pulse_match.group(1)

        # Diagnosis (Regex heuristic: looking for keywords near 'Diagnosis' or known diseases)
        diag_match = re.search(r'(?:Diagnosis|Assessment|Impression)\s*[:\-]\s*([^\n]+(?:\n\s*[\-\*o\d\.]+\s+[^\n]+)*)', text, re.IGNORECASE)
        if diag_match:
            # Split by commas or newlines with bullets
            items = re.split(r',| and |\n', diag_match.group(1))
            for d in items:
                clean_d = re.sub(r'^[\-\*o\d\.]+\s*', '', d).strip()
                if clean_d and clean_d.lower() not in ("diagnosis:", "assessment:", "impression:"):
                    data["diagnosis"].append(clean_d)

        # Medicines (Regex heuristic: looking for 'Rx', 'Medication', etc.)
        rx_block = re.search(r'(?:Rx|Medications?|Medicines?|Prescription)\s*[:\-]\s*(.*?)(?:Instructions?|Advice|Plan|BP|Blood Pressure|Temp|Temperature|Pulse|HR|Vitals|$)', text, re.IGNORECASE | re.DOTALL)
        if rx_block:
            rx_text = rx_block.group(1)
            lines = [l.strip() for l in rx_text.split('\n') if l.strip()]
            for line in lines:
                med_info = self._parse_medicine_line(line)
                if med_info:
                    data["medicines"].append(med_info)

        # Lines anywhere that start with a dosage form ("Tab. Dolo 650mg 1-0-1")
        # are medicines even without an Rx header or a dictionary match.
        seen = {m["name"].lower() for m in data["medicines"]}
        for line in text.split('\n'):
            bare = re.sub(r'^[\-\*•o\d\.\)]+\s*', '', line.strip())
            if DOSAGE_FORM_PREFIX.match(bare):
                med_info = self._parse_medicine_line(bare)
                if med_info and med_info["name"].lower() not in seen:
                    data["medicines"].append(med_info)
                    seen.add(med_info["name"].lower())

        # Instructions
        inst_match = re.search(r'(?:Instructions?|Advice|Plan|Follow up)\s*[:\-]\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if inst_match:
            inst_text = inst_match.group(1)
            lines = [l.strip() for l in inst_text.split('\n') if l.strip()]
            data["instructions"].extend(lines)

        return data

    def _parse_medicine_line(self, line: str) -> Optional[Dict]:
        """Parses a medication line to extract name, dosage, and frequency."""
        # Simple extraction logic based on common patterns
        dosage_match = DOSAGE_PATTERN.search(line)
        freq_match = FREQUENCY_PATTERN.search(line)
        
        dosage = dosage_match.group(1) if dosage_match else None
        freq = freq_match.group(1) if freq_match else None
        
        # Assume the medicine name is everything before the dosage
        name = line
        if dosage:
            name = name[:name.find(dosage)].strip()
        elif freq:
            name = name[:name.find(freq)].strip()
            
        # Clean up name
        name = re.sub(r'^[\-\*o\d\.]+\s*', '', name).strip() # Remove bullets/numbering
        name = DOSAGE_FORM_PREFIX.sub('', name).strip()  # Remove "Tab." / "Cap." etc.

        if len(name) > 2:
            return {
                "name": name,
                "dosage": dosage,
                "frequency": freq
            }
        return None

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extracts general entities using SpaCy."""
        data = {
            "persons": [],
            "orgs": [],
            "dates": []
        }
        
        if not self.nlp:
            return data
            
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    data["persons"].append(ent.text.strip())
                elif ent.label_ == "ORG":
                    data["orgs"].append(ent.text.strip())
                elif ent.label_ == "DATE":
                    data["dates"].append(ent.text.strip())
        except Exception as e:
            logger.exception("SpaCy extraction failed: %s", e)
            
        return data

    def _extract_with_scispacy(self, text: str) -> Dict[str, Any]:
        """Extracts medical entities using SciSpaCy."""
        data = {
            "chemicals": [],
            "diseases": []
        }
        
        if not self.sci_nlp:
            return data
            
        try:
            doc = self.sci_nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("CHEMICAL", "ENTITY"):
                    # Basic filter to avoid super short acronyms or non-words
                    clean_ent = ent.text.strip()
                    if len(clean_ent) > 3 and clean_ent.lower() != "opd":
                        data["chemicals"].append(clean_ent)
                elif ent.label_ == "DISEASE":
                    # Skip vitals the model mislabels as diseases, e.g. "100.4F"
                    if re.search(r'[A-Za-z]{3}', ent.text):
                        data["diseases"].append(ent.text.strip())
        except Exception as e:
            logger.exception("SciSpaCy extraction failed: %s", e)
            
        return data

    def _extract_from_dictionaries(self, text: str) -> Dict[str, Any]:
        """Extracts exact matches from the provided dictionaries."""
        text_lower = text.lower()
        
        found_drugs = []
        for drug in self.drugs:
            if " " not in drug:
                if re.search(r'\b' + re.escape(drug) + r'\b', text_lower):
                    found_drugs.append(drug)
            else:
                if drug in text_lower:
                    found_drugs.append(drug)
                    
        found_diseases = []
        for disease in self.diseases:
            if " " not in disease:
                if re.search(r'\b' + re.escape(disease) + r'\b', text_lower):
                    found_diseases.append(disease)
            else:
                if disease in text_lower:
                    found_diseases.append(disease)
                    
        return {
            "drugs": sorted(found_drugs, key=len, reverse=True), # Longest matches first
            "diseases": sorted(found_diseases, key=len, reverse=True)
        }

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Runs all extraction layers and merges the results into a structured JSON dict.
        """
        if not text or not text.strip():
            return {}

        # 1. Run all layers
        regex_data = self._extract_with_regex(text)
        spacy_data = self._extract_with_spacy(text)
        scispacy_data = self._extract_with_scispacy(text)
        dict_data = self._extract_from_dictionaries(text)

        # 2. Merge and Deduplicate
        final_data = regex_data.copy()

        # Merge Date
        if not final_data["date"] and spacy_data["dates"]:
            final_data["date"] = spacy_data["dates"][0]

        # Merge Hospital
        if not final_data["hospital"] and spacy_data["orgs"]:
            final_data["hospital"] = spacy_data["orgs"][0]

        # Merge Persons (try to guess Doctor vs Patient)
        if spacy_data["persons"]:
            for person in spacy_data["persons"]:
                if "Dr." in person or "dr." in person.lower() or "md" in person.lower():
                    if not final_data["doctor_name"]:
                        final_data["doctor_name"] = person
                else:
                    doctor = (final_data["doctor_name"] or "").lower()
                    if not final_data["patient_name"] and person.lower() not in doctor and not any(
                        part in doctor for part in person.lower().split()[:2] if len(part) > 2
                    ):
                        # Avoid replacing a regex-found patient name, but set if empty
                        final_data["patient_name"] = person

        # Merge Diagnosis
        diag_set = set(d.lower() for d in final_data["diagnosis"])
        
        # Add SciSpacy diseases (with substring check to prevent duplicates)
        for disease in scispacy_data["diseases"]:
            is_substring = any(disease.lower() in existing for existing in diag_set) or any(existing in disease.lower() for existing in diag_set)
            if not is_substring:
                final_data["diagnosis"].append(disease.title())
                diag_set.add(disease.lower())
                
        # Add Dictionary diseases (only if not already found to prevent substring matches from cluttering)
        for disease in dict_data["diseases"]:
            is_substring = any(disease.lower() in existing for existing in diag_set) or any(existing in disease.lower() for existing in diag_set)
            if not is_substring:
                final_data["diagnosis"].append(disease.title())
                diag_set.add(disease.lower())

        # Merge Medicines
        existing_meds_lower = set(m["name"].lower() for m in final_data["medicines"])
        
        # Process chemicals and dict drugs, trying to parse context around them to get dosage
        potential_meds = set(scispacy_data["chemicals"] + dict_data["drugs"])

        # Longest first, so "potassium chloride" is kept and "potassium" is skipped
        for med_name in sorted(potential_meds, key=len, reverse=True):
            # Check if already captured by regex
            is_substring = any(med_name.lower() in existing for existing in existing_meds_lower)
            if not is_substring:
                # Try to extract dosage/freq near this mention in the raw text
                # We'll grab a window of text around the first mention
                match = re.search(r'\b' + re.escape(med_name) + r'\b([^\n]{0,60})', text, re.IGNORECASE)
                dosage = None
                freq = None
                if match:
                    window = match.group(1)
                    dosage_match = DOSAGE_PATTERN.search(window)
                    freq_match = FREQUENCY_PATTERN.search(window)
                    if dosage_match: dosage = dosage_match.group(1)
                    if freq_match: freq = freq_match.group(1)
                
                # Only add if it has a dosage, OR if it's explicitly in our drug dictionary
                if dosage or (med_name.lower() in self.drugs):
                    final_data["medicines"].append({
                        "name": med_name.title(),
                        "dosage": dosage,
                        "frequency": freq
                    })
                    existing_meds_lower.add(med_name.lower())

        return final_data
