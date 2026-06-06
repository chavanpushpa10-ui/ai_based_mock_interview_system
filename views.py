from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from app.verify import authentication
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from django.contrib import messages
import PyPDF2
import joblib
from .models import Resume_data
import datetime
from moviepy.editor import VideoClip, AudioFileClip, VideoFileClip
import speech_recognition as sr
import io
import os
import moviepy.editor as mp
from django.conf import settings
from .prediction import *
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import time
import csv
import random
from django.urls import reverse
import difflib
import speech_recognition as sr



# Create your views here.
def convert_speech_to_text():
    """
    Use SpeechRecognition library to convert speech to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak your answer...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Request error from Google Speech Recognition service: {e}")
            return ""

def adminlogin(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        if email == 'user' and password == 'user':
            user = User.objects.filter(username='user').first()
            if not user:
                user = User.objects.create_user(username='user', password='user')
            login(request, user)
            messages.success(request, 'Login successful')
            return redirect('admindashboard')
        else:
            messages.error(request, 'Invalid credentials')
            return render(request, 'adminlogin.html')
    return render(request, 'adminlogin.html')

@login_required(login_url="adminlogin")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def admindashboard(request):
    users = User.objects.all()  # Fetch all users from the database
    return render(request, "admindashboard.html", {'users': users})

def delete_user(request, user_id):
    # Get the user by ID, excluding superuser or admin if needed
    user = get_object_or_404(User, id=user_id)
    
    # Ensure only certain users can be deleted
    if not user.is_superuser and user.username not in ["admin", "user"]:
        user.delete()
        messages.success(request, 'User deleted successfully.')
    else:
        messages.error(request, 'This user cannot be deleted.')
    
    # Redirect to the user records page (replace with the actual view)
    return redirect('admindashboard')  # Replace 'admindashboard' with the correct URL name

def index(request):
    return render(request,'index.html')

def test(request):
    return render(request,'test.html')

def about(request):
    return render(request,'about.html')

def services(request):
    return render(request,'services.html')

def contact(request):
    return render(request,'contact.html')

def resume(request):
    return render(request,'resume.html')

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html")

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['cpassword']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html")


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname' : request.user.first_name
    }
    return render(request, 'dashboard.html',context)

import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def resume(request):
    context = {
        'fname': request.user.first_name
    }
    predicted_category = None  # Initialize predicted_category variable
    if request.method == "POST":
        # Retrieve the uploaded resume file
        user_resume = request.FILES["user_resume"]
        # Check if a file was uploaded
        if user_resume:
            # Initialize an empty string to store text
            text = ''
            # Open the uploaded PDF file
            with user_resume.open() as pdf_file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Iterate through each page and extract text
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

                # Load the model, vectorizer, and encoder
                loaded_model = joblib.load('Models/resume_category_model.pkl')
                loaded_vectorizer = joblib.load('Models/resume_tfidf_vectorizer.pkl')
                loaded_encoder = joblib.load('Models/resume_label_encoder.pkl')

                cleaned_input = cleanResume(text)
                input_vectorized = loaded_vectorizer.transform([cleaned_input])
                prediction = loaded_model.predict(input_vectorized)
                predicted_category = loaded_encoder.inverse_transform(prediction)[0]
                user_level = level_identifier(len(pdf_reader.pages))
                actual_skills, recommended_skills = skills_having(text, predicted_category)
                resume_score = find_resume_score(text)
                
                data = Resume_data(user_first_name=request.user.first_name, user_last_name=request.user.last_name, user_email=request.user.username, user_resume=user_resume, cv_prediction=predicted_category, user_level=user_level, actual_skills=actual_skills, recommended_skills=recommended_skills, resume_score=resume_score, no_of_pages=len(pdf_reader.pages))
                data.date = datetime.date.today()
                data.save()

            # Pass the extracted text to the template
            messages.success(request, "Resume Uploaded Successfully!!!")
            return redirect('user_intro')  # Redirect to technicalinterview view with the predicted_category

    context['predicted_category'] = predicted_category  # Pass the predicted category to the template
    return render(request, 'resume.html', context)


from django.http import HttpResponseRedirect
from django.urls import reverse

@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def user_intro(request):
    intro_text = Resume_data.objects.last()
    if request.method == "POST":
        text = convert_speech_to_text()
        print("Text : ", text)
        loaded_model = joblib.load('Models/resume_category_model.pkl')
        loaded_vectorizer = joblib.load('Models/resume_tfidf_vectorizer.pkl')
        loaded_encoder = joblib.load('Models/resume_label_encoder.pkl')

        cleaned_input = cleanResume(text)
        print("Cleaned Text : ", cleaned_input)
        input_vectorized = loaded_vectorizer.transform([cleaned_input])
        prediction = loaded_model.predict(input_vectorized)
        predicted_category = loaded_encoder.inverse_transform(prediction)[0]
        print("Predicted Category : ", predicted_category)
        intro_text.intro_prediction = predicted_category
        intro_text.save()
        messages.success(request, "Audio Analyzed Successfully!")

        return redirect("technicalinterview")


    context = {
        'fname': request.user.first_name
    }
    return render(request, 'user_intro.html', context)


@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def result(request):
    final_result = Resume_data.objects.last()
    resume_url = str(final_result.user_resume)
    pdf_file_path = "media/" + resume_url
    skills = eval(final_result.actual_skills)
    rec_skills = eval(final_result.recommended_skills)

    overall_percentage = request.session.get('overall_percentage', 0)  # Default to 0 if not found

    # Retrieve the test score from the session
    test_score_percentage = request.session.get('test_score_percentage', 0)  # Default to 0 if not found

    context = {
        'final_result': final_result,
        'fname': request.user.first_name,
        'pdf_file_path': pdf_file_path,
        'skills': skills,
        'rec_skills': rec_skills,
        'overall_percentage': overall_percentage,
        'test_score_percentage': test_score_percentage,  # Add test score to context
    }
    return render(request, "result.html", context)


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")


def load_csv(file_path):
    """
    Load the CSV file containing questions, answers, and categories.
    """
    data = []
    try:
        with open(file_path, mode='r', encoding='latin1') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append({'question': row['Question'], 'answer': row['Answer'], 'category': row['Category']})
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
    return data

def filter_questions_by_category(data, category):
    """
    Filter questions by the selected category.
    """
    return [item for item in data if item['category'].lower() == category.lower()]


def calculate_match_percentage(actual_answer, user_answer):
    """
    Calculate the matching percentage between actual answer and user answer.
    """
    matcher = difflib.SequenceMatcher(None, actual_answer.lower(), user_answer.lower())
    return matcher.ratio() * 100

def technicalinterview(request):
    csv_file_path = "data set/Questions_Answers.csv"
    data = load_csv(csv_file_path)

    # Retrieve the predicted category from query parameters
    intro_text = Resume_data.objects.last()
    predicted_category = intro_text.intro_prediction

    if predicted_category:
        filtered_questions = filter_questions_by_category(data, predicted_category)
        questions_to_ask = random.sample(filtered_questions, min(5, len(filtered_questions)))
    else:
        questions_to_ask = []

    total_percentage = 0
    answered_questions = 0

    for idx, item in enumerate(questions_to_ask):
        print(f"\nQuestion {idx + 1}: {item['question']}")

        if request.method == "POST":
            user_answer = convert_speech_to_text()
            if not user_answer:
                print("No valid answer detected. Skipping...\n")
                continue

            match_percentage = calculate_match_percentage(item['answer'], user_answer)
            total_percentage += match_percentage
            answered_questions += 1

            print(f"Actual Answer: {item['answer']}")
            print(f"Your Answer: {user_answer}")
            print(f"Match Percentage: {match_percentage:.2f}%\n")

    overall_percentage = 0
    if answered_questions > 0:
        overall_percentage = total_percentage / answered_questions
        overall_percentage *= 2  

    # Store the overall_percentage in session
    request.session['overall_percentage'] = overall_percentage

    print("overall_percentage is : ", overall_percentage)

    context = {
        'overall_percentage': overall_percentage,
        'questions_to_ask': questions_to_ask,
        'predicted_category': predicted_category,
    }

    return render(request, "technicalinterview.html", context)

from django.shortcuts import render, redirect
import csv

# Path to your CSV file
file_path1 = 'data set/MCQQuestions.csv'

def load_csv_data(file_path1):
    questions_data = []
    try:
        with open(file_path1, mode='r', newline='', encoding='latin1') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Convert keys to avoid issues in templates
                formatted_row = {key.replace(' ', '_'): value.strip() for key, value in row.items()}
                questions_data.append(formatted_row)
        return questions_data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def user_test(request):
    # Get the last resume data object to determine the predicted category
    intro_text = Resume_data.objects.last()
    predicted_category = intro_text.intro_prediction if intro_text else None

    # Load the questions from the CSV
    questions_data = load_csv_data(file_path1)
    if not questions_data:
        return render(request, 'error.html', {'message': 'Failed to load questions.'})

    # Filter questions by the predicted category
    category_questions = [
        q for q in questions_data
        if q.get('Category', '').lower() == (predicted_category or '').lower()
    ]

    # Select the first 10 questions from the filtered questions
    selected_questions = category_questions[:10]

    if request.method == "POST":
        # Get user answers from the form
        user_answers = {}
        for i, question in enumerate(selected_questions):
            # Get the selected option for the current question
            user_answer = request.POST.get(f"Q{i+1}")
            if user_answer:
                # Get the question text (or any unique identifier for it)
                question_text = question.get('Questions', 'Unknown question')

                # Store the selected option along with the question text
                selected_option = question.get(f'option_{user_answer}', '').strip()
                user_answers[question_text] = {
                    'selected_option': selected_option,
                    'correct_answer': question.get('Correct_Answer', '').strip()
                }

        # Debugging: Print selected options with question texts
        print(user_answers)

        correct_answers = 0

        # Compare the selected answers with the correct answers
        for question_text, answer_data in user_answers.items():
            selected_option = answer_data['selected_option']
            correct_answer = answer_data['correct_answer']

            # Debugging: print selected option and correct answer for comparison
            print(f"Question: {question_text}")
            print(f"Selected option: {selected_option}, Correct answer: {correct_answer}")

            # Check if the selected option matches the correct answer
            if selected_option == correct_answer:
                correct_answers += 1

        # Calculate the score percentage
        score_percentage = (correct_answers / len(selected_questions)) * 100 if selected_questions else 0
        request.session['test_score_percentage'] = score_percentage

        # Redirect to the result page
        return redirect('result')

    # Pass the selected questions to the template for rendering
    return render(request, 'user_test.html', {'questions': selected_questions})










# def user_test(request):
#     # Retrieve the predicted category
#     intro_text = Resume_data.objects.last()
#     predicted_category = intro_text.intro_prediction if intro_text else None

#     # Load questions and filter by predicted category
#     questions_data = load_csv_data(file_path1)
#     category_questions = [q for q in questions_data if q.get('Category') == predicted_category]

#     # Select 10 random questions from the filtered list
#     random_questions = random.sample(category_questions, 10) if len(category_questions) >= 10 else category_questions

#     if request.method == "POST":
#         # Evaluate the user's answers
#         user_answers = {f"Q{i+1}": request.POST.get(f"Q{i+1}") for i in range(len(random_questions))}
#         correct_answers = 0
#         for i, question in enumerate(random_questions):
#             if user_answers.get(f"Q{i+1}") == question.get('Correct_Answer', ''):
#                 correct_answers += 1
#         score = (correct_answers / len(random_questions)) * 100

#         # Redirect with the score passed as a session variable
#         request.session['score'] = score
#         return redirect('result')  # Replace 'result' with the name of your URL pattern for the result page

#     return render(request, 'user_test.html', {'questions': random_questions, 'category': predicted_category})

# i want to print that score in below result views.py code

# @login_required(login_url="log_in")
# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
# def result(request):
#     final_result = Resume_data.objects.last()
#     resume_url = str(final_result.user_resume)
#     pdf_file_path = "media/" + resume_url
#     skills = eval(final_result.actual_skills)
#     rec_skills = eval(final_result.recommended_skills)

#     # Retrieve the overall_percentage from session
#     overall_percentage = request.session.get('overall_percentage', 0)  # Default to 0 if not found

#     context = {
#         'final_result': final_result,
#         'fname': request.user.first_name,
#         'pdf_file_path': pdf_file_path,
#         'skills': skills,
#         'rec_skills': rec_skills,
#         'overall_percentage': overall_percentage,  # Add overall_percentage to context
#     }
#     return render(request, "result.html", context)



