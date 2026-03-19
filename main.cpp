#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <queue>
#include <tuple>
#include <list>
#include <map>
#include <unordered_map>
#include <set>
#include <string>

int main() {
    std::cout << "vector" << std::endl;

    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    numbers[3] = 10;
    numbers.pop_back();
    numbers.insert(numbers.begin() + 1, 20);
    numbers.resize(10);

    std::vector<int> others = {100, 200, 300};
    numbers.insert(numbers.end(), others.begin(), others.end());

    // std::sort(numbers.begin(), numbers.end(), [](int a, int b) {
    //     return a > b;
    // });

    std::cout << numbers.size() << std::endl;

    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "deque" << std::endl;
    std::deque<int> deq;
    deq.push_back(1);
    deq.push_back(2);
    deq.push_back(3);
    std::cout << deq.front() << std::endl;
    deq.pop_front();
    std::cout << deq.size() << std::endl;   
    for (const auto& val : deq) {
        std::cout << val << " ";
    }
    std::cout << std::endl;


    std::cout << "priority_queue" << std::endl;
    std::priority_queue<int> pq;   // max heap
    pq.push(5);
    pq.push(1);
    pq.push(10);
    std::cout << pq.top() << std::endl;
    pq.pop();
    std::cout << pq.top() << std::endl;

    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq; // min heap
    min_pq.push(5);
    min_pq.push(1);
    min_pq.push(10);
    std::cout << min_pq.top() << std::endl;
    min_pq.pop();
    std::cout << min_pq.top() << std::endl;

    std::cout << "priority queue with tuple" << std::endl;
    using T = std::tuple<int, int, int>;

    struct Compare {
        bool operator()(const T& a, const T& b) {
            if (std::get<1>(a) == std::get<1>(b)) {
                return std::get<0>(a) > std::get<0>(b); // if second elements are equal, compare first elements
            }
            return std::get<1>(a) > std::get<1>(b); // min-heap based on second element
        }
    };

    std::priority_queue<T, std::vector<T>, Compare> tuple_pq;
    tuple_pq.push(std::make_tuple(1, 5, 100));
    tuple_pq.push(std::make_tuple(2, 3, 200));
    tuple_pq.push(std::make_tuple(3, 5, 50));

    while (!tuple_pq.empty()) {
        T top = tuple_pq.top();
        std::cout << "(" << std::get<0>(top) << ", " << std::get<1>(top) << ", " << std::get<2>(top) << ")" << std::endl;
        tuple_pq.pop();
    }

    // Tuples are mutable
    auto a = std::make_tuple(1, 2.5, "example");
    std::cout << "Tuple elements: "
              << std::get<0>(a) << ", "
              << std::get<1>(a) << ", "
              << std::get<2>(a) << std::endl;
    std::get<0>(a) = 10;
    std::cout << "Modified tuple elements: "
              << std::get<0>(a) << ", "
              << std::get<1>(a) << ", "
              << std::get<2>(a) << std::endl;

    std::cout << "list" << std::endl;
    std::list<int> lst = {1, 2, 3, 3, 3, 4, 5};
    lst.push_back(6);
    lst.push_front(0);
    for (const auto& val : lst) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    lst.remove(3); // remove all elements with value 3
    for (const auto& val : lst) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "map" << std::endl;
    std::map<std::string, int> age_map;
    age_map["Alice"] = 30;
    age_map["Bob"] = 25;
    age_map["Charlie"] = 35;
    for (const auto& pair : age_map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    std::cout << "unordered_map" << std::endl;
    std::unordered_map<std::string, int> score_map;
    score_map["Math"] = 95;
    score_map["Science"] = 90;
    score_map["English"] = 85;
    for (const auto& pair : score_map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    std::cout << "multiset" << std::endl;
    std::multiset<int> ms;
    ms.insert(5);
    ms.insert(1);
    ms.insert(5);
    ms.insert(3);
    ms.insert(10);
    ms.insert(5);
    for (const auto& val : ms) {
        std::cout << val << " ";
    }
    std::cout << std::endl; 

    auto lb = ms.lower_bound(5);
    auto ub = ms.upper_bound(5);
    --lb;
    std::cout << *lb << std::endl; // largest element less than 5
    ++lb;
    std::cout << *lb << std::endl;
    // ++ub;
    std::cout << *ub << std::endl; 
    --ub;
    std::cout << *ub << std::endl; // largest element equal to 5

    auto lb2 = ms.lower_bound(8);
    std::cout << *lb2 << std::endl; // first element greater than or equal to 8

    std::cout << "test" << std::endl;
    std::vector<std::vector<int>> vec2d;
    vec2d.push_back({1, 2, 3});
    vec2d.push_back({4, 5, 6});
    for (const auto& row : vec2d) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::map<char, int>> trie;
    trie.push_back({}); // root node


    std::cout << "string" << std::endl;
    std::string str = "Hello, World!";

    std::cout << str.size() << std::endl;

    // Get 3-7 of the string
    std::string substr = str.substr(3, 5);
    std::cout << substr << std::endl;
    // Check if str starts with "Hello"
    if (str.find("Hello") == 0) {
        std::cout << "String starts with 'Hello'" << std::endl;
    }

    // Check if str ends with "Hello"
    if (str.find("Hello") == str.size() - 5) {
        std::cout << "String ends with 'Hello'" << std::endl;
    }
    else {
        std::cout << "String does not end with 'Hello'" << std::endl;
    }

    // Check if str contains "or"
    if (str.find("or") != std::string::npos) {
        std::cout << "String contains 'or'" << std::endl;
    }

    str.replace(7, 5, "Universe");
    std::cout << str << std::endl;

    str.erase(5, 2);
    std::cout << str << std::endl;

    std::string s;
    s += "ABCDE";
    s += "12345";
    std::cout << s << std::endl;
    std::cout << s + "XYZ" << std::endl;

    return 0;
}