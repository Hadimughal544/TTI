from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from .models import GeneratedImage
from .services import diffusion_service
import uuid
import os

def index(request):
    images = GeneratedImage.objects.all().order_by('-created_at')[:10]
    return render(request, 'generator/index.html', {'recent_images': images})

def generate(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        if not prompt:
            messages.error(request, "Please provide a prompt.")
            return redirect('index')

        try:
            # Generate image data
            image_data = diffusion_service.generate(prompt)
            
            # Create the model instance
            gen_image = GeneratedImage(prompt=prompt)
            
            # Create a filename
            filename = f"{uuid.uuid4().hex}.png"
            
            # Save the file to the model
            gen_image.image.save(filename, ContentFile(image_data), save=True)
            
            messages.success(request, "Image generated successfully!")
            return render(request, 'generator/index.html', {
                'generated_image': gen_image,
                'recent_images': GeneratedImage.objects.all().order_by('-created_at')[:10]
            })
        except Exception as e:
            messages.error(request, f"Failed to generate image: {str(e)}")
    
    return redirect('index')

def delete_image(request, image_id):
    image = get_object_or_404(GeneratedImage, id=image_id)
    try:
        # Delete the file from filesystem
        if image.image:
            if os.path.isfile(image.image.path):
                os.remove(image.image.path)
        
        # Delete the database record
        image.delete()
        messages.success(request, "Image deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting image: {str(e)}")
    
    return redirect('index')
