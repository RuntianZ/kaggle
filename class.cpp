#include <iostream>
#include <vector>
using namespace std;

class Person {
    protected:
        string name;
        int age;
    public:
        Person(const string& name, int age) : name(name), age(age) {}
        virtual void introduce() const {
            cout << "Name: " << name << ", Age: " << age << endl;
        }
};


class Student : public Person {
    private:
        string major;
    public:
        Student(const string& name, int age, const string& major)
            : Person(name, age), major(major) {}
        virtual void introduce() const override {
            Person::introduce();
            cout << "Major: " << major << endl;
        }
};


class MedicalStudent : public Student {
    private:
        int year;
    public:
        MedicalStudent(const string& name, int age, const string& major, int year)
            : Student(name, age, major), year(year) {}
        void introduce() const final override {
            Student::introduce();
            cout << "Year: " << year << endl;
        }
};

void test_introduction(const Person& person) {
    person.introduce();
}

int main() {
    Person person("John Doe", 40);
    Student student("Jane Smith", 20, "Computer Science");
    MedicalStudent med_student("Alice Johnson", 22, "Medicine", 3);

    test_introduction(person);
    test_introduction(student);
    test_introduction(med_student);

    return 0;
}