import java.util.*;


public class Test {
    public static void main(String[] args) {
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        deque.add(1);
        deque.add(2);
        deque.add(3);
        System.out.println(deque); // Output: [1, 2, 3]
        deque.addFirst(0);
        System.out.println(deque); // Output: [0, 1, 2, 3]
        deque.removeFirst();
        System.out.println(deque); // Output: [1, 2, 3]
        deque.removeLast();
        System.out.println(deque); // Output: [1, 2]

        HashSet<String> set = new HashSet<>();
        set.add("apple");
        set.add("banana");
        set.add("orange");
        System.out.println(set); // Output: [banana, orange, apple]
        set.add("banana"); // Duplicate, will not be added
        System.out.println(set); // Output: [banana, orange, apple]
        set.remove("orange");
        System.out.println(set); // Output: [banana, apple]

        HashMap<String, Integer> map = new HashMap<>();
        map.put("one", 1);
        map.put("two", 2);
        map.put("three", 3);
        System.out.println(map); // Output: {one=1, two=2, three=3}
        map.put("two", 22); // Update value for key "two"
        System.out.println(map); // Output: {one=1, two=22, three=3}
        map.remove("one");
        System.out.println(map); // Output: {two=22, three=3}
        for (String key : map.keySet()) {
            System.out.println(key + ": " + map.get(key));
        }
    }
}