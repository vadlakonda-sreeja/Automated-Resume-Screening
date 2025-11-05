from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})



def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})



def Training(request):
    from .utility import predict
    knn_report = predict.training_knn()
    nb_report = predict.training_NB()
    svm_report = predict.training_SVM()
    return render(request,'users/accuracy.html',{'nb': nb_report,"knn":knn_report, 'svm': svm_report})


def prediction(request):
    if request.method=='POST':
        from .utility import predict
        joninfo  = request.POST.get('joninfo')
        print('*'*50,joninfo)
        result = predict.prediction(joninfo)
        print(result)
        return render(request, 'users/Predication.html', {'result': result})
    else:
        
        return render(request,'users/Predication.html',{})
    

def UserImageTest(request):
    if request.method == 'POST':
        # myfile = request.FILES['file']
        # fs = FileSystemStorage(location = 'media/test_pdf')
        # filename = fs.save(myfile.name, myfile)
        # print(filename*2)
        # uploaded_file_url = fs.url(filename)
        from .utility import predict
        # result = weaponr_predictions.start_prediction(filename)
        filename = r'media\Mata Peddanna.pdf'
        result = f'Congrats your resume is selected for a {predict.pdf(filename)} profile'
        print('result', result)
        return render(request, "users/test_form.html", {"result": result})
    else:
        return render(request, "users/test_form.html", {})
    











     
    