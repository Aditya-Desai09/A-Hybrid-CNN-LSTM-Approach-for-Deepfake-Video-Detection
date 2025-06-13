from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.http import JsonResponse
from django.conf import settings
import os, shutil
import torch
import numpy as np
import cv2
from .inference_utils import run_inference, extract_faces, extract_suspicious_frames

# ---------------------- AUTH & STATIC ----------------------
def home(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def terms_view(request):
    return render(request, 'terms.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            next_url = request.GET.get('next') or 'upload'
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')
    return render(request, 'login.html')

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('register')
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('register')
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('register')

        user = User.objects.create_user(username=username, email=email, password=password)
        messages.success(request, "Registration successful. Please log in.")
        return redirect('login')

    return render(request, 'register.html')

# ---------------------- VIDEO UPLOAD & INFERENCE ----------------------
@login_required
def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('file')
        if not video_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        video_save_path = os.path.join(upload_path, video_file.name)

        with open(video_save_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Clean old frames/faces
        for folder in ['frames', 'faces']:
            folder_path = os.path.join(settings.MEDIA_ROOT, folder)
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path, exist_ok=True)

        label, confidence = run_inference(video_save_path)
        extract_suspicious_frames(video_save_path)
        extract_faces(video_save_path)

        request.session['video_name'] = video_file.name
        request.session['result_label'] = label
        request.session['confidence_score'] = confidence

        return JsonResponse({'redirect': '/result/'})
    return render(request, 'upload.html')

# ---------------------- RESULTS ----------------------
@login_required
def result_view(request):
    video_name = request.session.get('video_name', 'sample_video.mp4')
    label = request.session.get('result_label', 'UNCERTAIN')
    confidence = request.session.get('confidence_score', 0.0)

    frame_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
    face_dir = os.path.join(settings.MEDIA_ROOT, 'faces')

    frame_paths = ['media/frames/' + f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
    face_paths = ['media/faces/' + f for f in os.listdir(face_dir) if f.endswith('.jpg')]

    context = {
        'video_path': f'media/uploads/{video_name}',
        'result_label': label,
        'confidence_score': confidence,
        'frame_paths': frame_paths,
        'face_paths': face_paths,
    }
    return render(request, 'result.html', context)

# ---------------------- ACCURACY CHART PAGE ----------------------

def chart_view(request):
    return render(request, 'chart.html')
