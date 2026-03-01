"""End-to-end API tests for AutismCare Multi-Modal Backend."""
import requests
import json
import sys

API = "http://localhost:8000"

def test(name, fn):
    print(f"\n=== {name} ===")
    try:
        fn()
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    return True

results = []

def t1():
    r = requests.get(f"{API}/api/status")
    d = r.json()
    print(f"  Models: {json.dumps(d['models'])}")
    assert d["models"]["face_classifier"] == True
    assert d["models"]["questionnaire_xgb"] == True
    assert d["models"]["pose_skeleton_xgb"] == True
    assert d["models"]["eye_tracking_xgb"] == True

results.append(test("API Status", t1))

def t2():
    r = requests.get(f"{API}/api/model-info")
    d = r.json()
    print(f"  Keys: {list(d.keys())}")
    assert "questionnaire" in d
    assert "pose_skeleton" in d

results.append(test("Model Info", t2))

def t3():
    img_path = r"D:\Autism\AutismData\Autism Spectrum Detection ( from kaggle + zenodo )\ASD Data\ASD Data\Test\autism\001.jpg"
    with open(img_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        r = requests.post(f"{API}/api/analyze", files=files)
    d = r.json()
    print(f"  Face: {d['face_score']}, Fused: {d['fused_score']}")
    print(f"  Risk: {d['screening']['state']}")
    print(f"  GradCAM: {d['gradcam_b64'] is not None and len(d.get('gradcam_b64','')) > 100}")
    print(f"  Therapy items: {len(d['therapy']['plan'])}")
    assert r.status_code == 200
    assert "face_score" in d
    assert "screening" in d
    assert "clinical" in d
    assert "therapy" in d
    assert "monitoring" in d
    assert d["modality_scores"]["face"] is not None

results.append(test("Image Analysis", t3))

def t4():
    payload = {
        "A1_Score": 0, "A2_Score": 0, "A3_Score": 0, "A4_Score": 0, "A5_Score": 0,
        "A6_Score": 0, "A7_Score": 0, "A8_Score": 0, "A9_Score": 0, "A10_Score": 0,
        "age": 4.0, "gender": 1, "jundice": 0, "austim": 1
    }
    r = requests.post(f"{API}/api/questionnaire", json=payload)
    d = r.json()
    print(f"  Quest score: {d['questionnaire_score']}")
    print(f"  Domain: {d['domain_profile']}")
    print(f"  Risk: {d['screening']['state']}")
    print(f"  Total: {d['domain_scores']['total_score']}")
    assert r.status_code == 200
    assert d["questionnaire_score"] < 0.5  # all 0s = no ASD traits = low risk
    assert d["screening"]["state"] == "LOW_RISK"

results.append(test("Questionnaire (High Risk)", t4))

def t5():
    payload = {
        "A1_Score": 1, "A2_Score": 1, "A3_Score": 1, "A4_Score": 1, "A5_Score": 1,
        "A6_Score": 1, "A7_Score": 1, "A8_Score": 1, "A9_Score": 1, "A10_Score": 1,
        "age": 5.0, "gender": 0, "jundice": 0, "austim": 0
    }
    r = requests.post(f"{API}/api/questionnaire", json=payload)
    d = r.json()
    print(f"  Quest score: {d['questionnaire_score']}")
    print(f"  Risk: {d['screening']['state']}")
    assert r.status_code == 200
    assert d["questionnaire_score"] > 0.5  # all 1s = all ASD traits = high risk

results.append(test("Questionnaire (Low Risk)", t5))

def t6():
    r = requests.post(f"{API}/api/fuse", json={
        "face": 0.75, "questionnaire": 0.85, "eye_tracking": 0.60, "pose": 0.90
    })
    d = r.json()
    print(f"  Fused: {d['fused_score']}")
    print(f"  Risk: {d['screening']['state']}")
    print(f"  Flagged: {d['screening'].get('flagged_modalities', [])}")
    print(f"  Agreement: {d['screening'].get('cross_modal_agreement')}")
    assert r.status_code == 200
    assert d["fused_score"] > 0.5
    assert len(d["screening"].get("flagged_modalities", [])) > 0

results.append(test("Multi-Modal Fusion", t6))

def t7():
    r = requests.get(f"{API}/api/history")
    d = r.json()
    print(f"  Sessions: {d['sessions']}")
    print(f"  History len: {len(d['score_history'])}")
    assert d["sessions"] >= 3  # tests that add to history

results.append(test("History", t7))

def t8():
    r = requests.post(f"{API}/api/clear")
    d = r.json()
    assert d["status"] == "cleared"
    r2 = requests.get(f"{API}/api/history")
    assert r2.json()["sessions"] == 0

results.append(test("Clear History", t8))

# Summary
print("\n" + "=" * 50)
passed = sum(results)
total = len(results)
print(f"  Results: {passed}/{total} tests passed")
if passed == total:
    print("  ALL TESTS PASSED!")
else:
    print("  SOME TESTS FAILED")
print("=" * 50)
