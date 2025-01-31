import json
import random
from typing import List, Dict

class CodingQuestionGenerator:
    def __init__(self):
        self.used_questions = set()  # To prevent duplicates
        
        # Define topic weights to ensure balanced distribution
        self.topics = {
            "data_structures": 0.2,
            "algorithms": 0.2,
            "system_design": 0.15,
            "basic_programming": 0.1,
            "object_oriented": 0.1,
            "database": 0.1,
            "web_development": 0.05,
            "concurrency": 0.05,
            "security": 0.03,
            "testing": 0.02
        }
        
        # Template generators for each topic
        self.templates = {
            "data_structures": self._generate_ds_question,
            "algorithms": self._generate_algo_question,
            "system_design": self._generate_system_design_question,
            "basic_programming": self._generate_basic_question,
            "object_oriented": self._generate_oop_question,
            "database": self._generate_db_question,
            "web_development": self._generate_web_question,
            "concurrency": self._generate_concurrency_question,
            "security": self._generate_security_question,
            "testing": self._generate_testing_question
        }

    def _generate_ds_question(self) -> Dict:
        structures = ["Array", "LinkedList", "Tree", "Graph", "Hash Table", "Stack", "Queue", "Heap"]
        operations = ["implement", "optimize", "modify", "balance", "traverse", "search in", "sort"]
        constraints = [
            "with O(1) space complexity",
            "with O(n) time complexity",
            "using only iterative approach",
            "using recursive approach",
            "handling edge cases",
            "with minimal memory usage"
        ]
        
        structure = random.choice(structures)
        operation = random.choice(operations)
        constraint = random.choice(constraints)
        
        question = f"{operation.capitalize()} a {structure} {constraint}."
        
        difficulty = random.choice(["medium", "hard"])
        
        return {
            "input": {
                "question": question,
                "difficulty": difficulty,
                "topic": "Data Structures",
                "tags": [structure.lower(), operation.split()[0], "optimization"]
            }
        }

    def _generate_algo_question(self) -> Dict:
        paradigms = ["Dynamic Programming", "Greedy", "Divide and Conquer", "Backtracking"]
        problem_types = [
            "Find the shortest path",
            "Calculate the minimum cost",
            "Find all possible combinations",
            "Optimize the solution",
            "Find the maximum profit",
            "Detect a cycle",
            "Find the longest sequence"
        ]
        
        paradigm = random.choice(paradigms)
        problem = random.choice(problem_types)
        
        conditions = [
            "given certain constraints",
            "in an efficient manner",
            "with optimal space usage",
            "handling multiple edge cases",
            "with minimal time complexity"
        ]
        
        question = f"{problem} using {paradigm} approach {random.choice(conditions)}."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Algorithms",
                "tags": [paradigm.lower().replace(" ", "_"), "optimization", "problem_solving"]
            }
        }

    def _generate_system_design_question(self) -> Dict:
        systems = [
            "URL shortener",
            "Social media feed",
            "Payment processing system",
            "Chat application",
            "File sharing service",
            "Video streaming platform",
            "Email service",
            "Shopping cart system"
        ]
        
        aspects = [
            "scalability",
            "reliability",
            "performance",
            "security",
            "data consistency",
            "fault tolerance"
        ]
        
        system = random.choice(systems)
        aspect = random.choice(aspects)
        
        question = f"Design a {system} with focus on {aspect}. Consider the trade-offs and explain your architectural decisions."
        
        return {
            "input": {
                "question": question,
                "difficulty": "hard",
                "topic": "System Design",
                "tags": ["architecture", "scalability", aspect]
            }
        }

    def _generate_basic_question(self) -> Dict:
        concepts = [
            "variable scoping",
            "error handling",
            "file I/O",
            "string manipulation",
            "array operations",
            "control structures",
            "functions",
            "recursion"
        ]
        
        tasks = [
            "Write a program to",
            "Implement a function that",
            "Create a solution for",
            "Develop a program that",
            "Write code to"
        ]
        
        operations = [
            "reverse a string",
            "find common elements in arrays",
            "check for palindromes",
            "calculate factorial",
            "handle file operations",
            "implement basic calculator",
            "validate input data"
        ]
        
        concept = random.choice(concepts)
        task = random.choice(tasks)
        operation = random.choice(operations)
        
        question = f"{task} {operation} using proper {concept}."
        
        return {
            "input": {
                "question": question,
                "difficulty": "easy",
                "topic": "Basic Programming",
                "tags": [concept.replace(" ", "_"), "fundamentals"]
            }
        }

    def _generate_oop_question(self) -> Dict:
        patterns = [
            "Singleton",
            "Factory",
            "Observer",
            "Strategy",
            "Decorator",
            "Command",
            "Adapter"
        ]
        
        scenarios = [
            "logging system",
            "payment processing",
            "game engine",
            "inventory management",
            "user authentication",
            "file system",
            "notification service"
        ]
        
        pattern = random.choice(patterns)
        scenario = random.choice(scenarios)
        
        question = f"Implement a {scenario} using the {pattern} design pattern. Ensure proper encapsulation and inheritance principles."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Object-Oriented Programming",
                "tags": ["design_patterns", pattern.lower(), "oop"]
            }
        }

    def _generate_db_question(self) -> Dict:
        operations = [
            "optimize a query",
            "design a schema",
            "implement indexing",
            "handle transactions",
            "ensure data consistency",
            "implement CRUD operations",
            "manage relationships"
        ]
        
        scenarios = [
            "e-commerce platform",
            "social media application",
            "banking system",
            "hotel booking service",
            "library management system",
            "student database",
            "inventory system"
        ]
        
        operation = random.choice(operations)
        scenario = random.choice(scenarios)
        
        question = f"{operation.capitalize()} for a {scenario}. Consider performance and scalability requirements."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Database",
                "tags": ["sql", "database_design", "optimization"]
            }
        }

    def _generate_web_question(self) -> Dict:
        topics = [
            "RESTful API",
            "authentication",
            "state management",
            "routing",
            "caching",
            "security",
            "performance optimization"
        ]
        
        frameworks = [
            "React",
            "Node.js",
            "Django",
            "Flask",
            "Spring Boot",
            "Express.js",
            "Angular"
        ]
        
        topic = random.choice(topics)
        framework = random.choice(frameworks)
        
        question = f"Implement {topic} in a {framework} application. Focus on best practices and performance."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Web Development",
                "tags": ["web", framework.lower(), topic.lower().replace(" ", "_")]
            }
        }

    def _generate_concurrency_question(self) -> Dict:
        concepts = [
            "thread synchronization",
            "deadlock prevention",
            "race condition handling",
            "parallel processing",
            "async/await implementation",
            "producer-consumer problem",
            "reader-writer problem"
        ]
        
        scenarios = [
            "bank transaction system",
            "ticket booking platform",
            "resource sharing system",
            "data processing pipeline",
            "concurrent file access",
            "message queue implementation"
        ]
        
        concept = random.choice(concepts)
        scenario = random.choice(scenarios)
        
        question = f"Implement a solution for {concept} in a {scenario}. Ensure thread safety and optimal performance."
        
        return {
            "input": {
                "question": question,
                "difficulty": "hard",
                "topic": "Concurrency",
                "tags": ["multithreading", "synchronization", "parallel_processing"]
            }
        }

    def _generate_security_question(self) -> Dict:
        topics = [
            "input validation",
            "authentication",
            "authorization",
            "encryption",
            "session management",
            "secure communication",
            "data protection"
        ]
        
        contexts = [
            "web application",
            "mobile app",
            "API endpoint",
            "database access",
            "file system",
            "network protocol",
            "user management system"
        ]
        
        topic = random.choice(topics)
        context = random.choice(contexts)
        
        question = f"Implement secure {topic} for a {context}. Address common vulnerabilities and follow security best practices."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Security",
                "tags": ["security", "best_practices", topic.replace(" ", "_")]
            }
        }

    def _generate_testing_question(self) -> Dict:
        testing_types = [
            "unit tests",
            "integration tests",
            "end-to-end tests",
            "performance tests",
            "security tests",
            "load tests",
            "stress tests"
        ]
        
        components = [
            "authentication service",
            "payment processor",
            "data validator",
            "API endpoint",
            "database connector",
            "caching system",
            "file handler"
        ]
        
        test_type = random.choice(testing_types)
        component = random.choice(components)
        
        question = f"Write comprehensive {test_type} for a {component}. Include edge cases and error scenarios."
        
        return {
            "input": {
                "question": question,
                "difficulty": random.choice(["medium", "hard"]),
                "topic": "Testing",
                "tags": ["testing", test_type.split()[0], "quality_assurance"]
            }
        }

    def generate_questions(self, count: int) -> List[Dict]:
        questions = []
        
        while len(questions) < count:
            # Select topic based on weights
            topic = random.choices(
                list(self.topics.keys()),
                weights=list(self.topics.values()),
                k=1
            )[0]
            
            # Generate question using appropriate template
            question = self.templates[topic]()
            
            # Convert to string for duplicate checking
            question_str = json.dumps(question, sort_keys=True)
            
            # Only add if it's not a duplicate
            if question_str not in self.used_questions:
                self.used_questions.add(question_str)
                questions.append(question)
        
        return questions

def main():
    # Initialize generator
    generator = CodingQuestionGenerator()
    
    # Generate 5000 questions
    questions = generator.generate_questions(1000)
    
    # Save to JSON file
    with open('data/codingQues.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(questions)} unique questions and saved to codingQues.json")

if __name__ == "__main__":
    main()