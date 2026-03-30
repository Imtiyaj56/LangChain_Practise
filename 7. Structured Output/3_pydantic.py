#Data Validation Using Pydantic Library

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

#schema
class Student(BaseModel):
    name : str = 'imtiyaj'   #default value = imtiyaj
    age : Optional[int] = None
    email : EmailStr  
    cgpa :  float = Field(gt=0, lt=10, default=5, description='It represent marks of students')  

new_student = {'name': 'raj', 'age': 21, 'email': 'abc@gecpalanpur.ac.in', 'cgpa':8}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict)

student_json = student.model_dump_json()
print(student_json)