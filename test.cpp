#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <map>
using namespace std;

int main() {

    vector<int> vec = {4, 2, 5, 2, 1, 4, 3};
    sort(vec.begin(), vec.end());
    vec.erase(unique(vec.begin(), vec.end()), vec.end());
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << endl;

    vector<map<int, int>> vec_map(5);
    vec_map[0][1] = 10;
    vec_map[1][2] = 20;
    for (const auto& m : vec_map) {
        for (const auto& p : m) {
            cout << p.first << ":" << p.second << " ";
        }
        cout << endl;
    }

    int a[5] = {};
    for (int i = 0; i < 5; ++i) {
        cout << a[i] << " ";
    }
    cout << endl;

    int b[5] = {0, 1, 2};
    for (int i = 0; i < 5; ++i) {
        cout << b[i] << " ";
    }
    cout << endl;

    string s = "abcdefabcdef";
    cout << s.find('b') << endl;
    cout << s.find('g') << endl;
    cout << s.find('d', 5) << endl;
    cout << s.substr(0, s.find('d')) << endl;
    cout << s.substr(0, s.find('g')) << endl;

    string s1 = "abcdefaaaaaa";
    s1.erase(6, string::npos);
    cout << s1 << endl;

    string s11 = "bcdeff";
    cout << s1 << ", " << s11 << ": " << s1.compare(s11) << endl;

    string s3 = "/a/b/cc/d/eeee";
    cout << s3.find('/', 1) << endl;
    cout << s3.find('/', 10) << endl;

    vector<int> v = {1, 2, 3, 4, 5};
    v.resize(3);
    for (const auto& val : v) {
        cout << val << " ";
    }
    cout << endl;

    string s2 = "123";
    int num = stoi(s2);
    cout << num + 1 << endl;

    int myints[] = {75,23,65,42,13};
    std::set<int> myset (myints,myints+5);

    std::cout << "myset contains:";
    for (std::set<int>::iterator it=myset.begin(); it!=myset.end(); ++it)
        std::cout << ' ' << *it;

    std::cout << '\n';
    return 0;
}