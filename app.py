import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clf_model = joblib.load('artifacts/best_model_clf.pkl')
reg_model = joblib.load('artifacts/best_model_reg.pkl')

def main():
    st.set_page_config(page_title="Student Placement Predictor", layout="wide")

    st.sidebar.title("Student Placement App")
    st.sidebar.markdown("Prediksi status penempatan kerja dan estimasi gaji mahasiswa berdasarkan data akademik dan skill.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Panduan Penggunaan")
    st.sidebar.markdown("""
    1. Isi seluruh data mahasiswa pada form
    2. Klik tombol **Make Prediction**
    3. Lihat hasil prediksi dan estimasi gaji
    """)
    st.sidebar.markdown("---")

    st.title("Student Placement Prediction")
    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("Data Akademik")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            ssc_percentage = st.number_input("SSC Percentage (%)", 30, 100)
            hsc_percentage = st.number_input("HSC Percentage (%)", 30, 100)
            degree_percentage = st.number_input("Degree Percentage (%)", 30, 100)
            cgpa = st.number_input("CGPA", 0.0, 10.0, value=7.5, step=0.01)
            entrance_exam_score = st.number_input("Entrance Exam Score", 0, 100)
            attendance_percentage = st.number_input("Attendance Percentage (%)", 0, 100)
            backlogs = st.number_input("Backlogs", 0, 20)

        with col2:
            st.subheader("Skills & Experience")
            technical_skill_score = st.number_input("Technical Skill Score", 0, 100)
            soft_skill_score = st.number_input("Soft Skill Score", 0, 100)
            internship_count = st.number_input("Internship Count", 0, 10)
            live_projects = st.number_input("Live Projects", 0, 20)
            work_experience_months = st.number_input("Work Experience (in months)", 0, 60)
            certifications = st.number_input("Certifications", 0, 20)
            extracurricular_activities = st.radio("Extracurricular Activities", ["Yes", "No"])

        submitted = st.form_submit_button("Make Prediction", use_container_width=True)

    if submitted:
        data = {
            'gender': gender,
            'ssc_percentage': int(ssc_percentage),
            'hsc_percentage': int(hsc_percentage),
            'degree_percentage': int(degree_percentage),
            'cgpa': float(cgpa),
            'entrance_exam_score': int(entrance_exam_score),
            'technical_skill_score': int(technical_skill_score),
            'soft_skill_score': int(soft_skill_score),
            'internship_count': int(internship_count),
            'live_projects': int(live_projects),
            'work_experience_months': int(work_experience_months),
            'certifications': int(certifications),
            'attendance_percentage': int(attendance_percentage),
            'backlogs': int(backlogs),
            'extracurricular_activities': extracurricular_activities,
        }

        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

        placement_pred = clf_model.predict(df)[0]
        placement_prob = clf_model.predict_proba(df)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            if placement_pred == 1:
                st.success("Placement Prediction: Placed")
                salary_pred = reg_model.predict(df)[0]
                st.success(f"Estimated Salary: {salary_pred:.2f} LPA")
            else:
                st.error("Placement Prediction: Not Placed")
                st.info("Estimated Salary: 0.00 LPA")

            st.metric("Probability Placed", f"{placement_prob[1]*100:.1f}%")
            st.metric("Probability Not Placed", f"{placement_prob[0]*100:.1f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            labels = ["Not Placed", "Placed"]
            colors = ["#e74c3c", "#2ecc71"]
            ax.bar(labels, placement_prob, color=colors)
            ax.set_ylabel("Probability")
            ax.set_title("Placement Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
