"""MMLU-style multiple-choice knowledge data (keeps general knowledge and the answer-letter habit intact while the model is fine-tuned on agent data).

    python -m distill.mcq_gen --n 3600 --out data/raw/mcq_knowledge.jsonl

Facts come from hand-checked tables (capitals, elements, authors, discoveries, SI units, planets, history dates, computing, biology, chemistry, physics,
health, economics) and from COMPUTED arithmetic / number-theory questions, so every correct option is right by construction and distractors are drawn
from the same table (plausible but wrong). Prompts use several MMLU-like layouts; answers are either the letter alone, "The answer is C", or a short
reasoning (recall the fact, check the option, state the letter) ending in "Final answer: C". About 18% are two-turn: after the answer the user asks why
the other options are wrong, and the explanation is built from the table rows of those distractors.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

LET = "ABCD"

CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
            ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Greece", "Athens"), ("Ireland", "Dublin"), ("Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"), ("Czech Republic", "Prague"), ("Hungary", "Budapest"), ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "New Delhi"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
            ("Nigeria", "Abuja"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Turkey", "Ankara"),
            ("Russia", "Moscow"), ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Morocco", "Rabat"), ("Mexico", "Mexico City"), ("Colombia", "Bogota"), ("Cuba", "Havana"),
            ("New Zealand", "Wellington")]
ELEMENTS = [("Hydrogen", "H", 1), ("Helium", "He", 2), ("Lithium", "Li", 3), ("Beryllium", "Be", 4), ("Boron", "B", 5), ("Carbon", "C", 6), ("Nitrogen", "N", 7), ("Oxygen", "O", 8),
            ("Fluorine", "F", 9), ("Neon", "Ne", 10), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12), ("Aluminium", "Al", 13), ("Silicon", "Si", 14), ("Phosphorus", "P", 15), ("Sulfur", "S", 16),
            ("Chlorine", "Cl", 17), ("Argon", "Ar", 18), ("Potassium", "K", 19), ("Calcium", "Ca", 20), ("Iron", "Fe", 26), ("Copper", "Cu", 29), ("Zinc", "Zn", 30), ("Tin", "Sn", 50),
            ("Silver", "Ag", 47), ("Gold", "Au", 79), ("Mercury", "Hg", 80), ("Lead", "Pb", 82)]
AUTHORS = [("Hamlet", "William Shakespeare"), ("Pride and Prejudice", "Jane Austen"), ("1984", "George Orwell"), ("War and Peace", "Leo Tolstoy"), ("Don Quixote", "Miguel de Cervantes"),
           ("The Odyssey", "Homer"), ("Moby-Dick", "Herman Melville"), ("Great Expectations", "Charles Dickens"), ("The Divine Comedy", "Dante Alighieri"), ("Crime and Punishment", "Fyodor Dostoevsky"),
           ("Frankenstein", "Mary Shelley"), ("The Hobbit", "J. R. R. Tolkien"), ("Les Miserables", "Victor Hugo"), ("The Trial", "Franz Kafka"), ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
           ("Ulysses", "James Joyce"), ("Jane Eyre", "Charlotte Bronte"), ("Wuthering Heights", "Emily Bronte"), ("The Great Gatsby", "F. Scott Fitzgerald"), ("To Kill a Mockingbird", "Harper Lee"),
           ("Brave New World", "Aldous Huxley"), ("The Stranger", "Albert Camus"), ("Faust", "Johann Wolfgang von Goethe"), ("Madame Bovary", "Gustave Flaubert")]
DISCOVERIES = [("discovered penicillin", "Alexander Fleming"), ("invented the first practical telephone", "Alexander Graham Bell"), ("formulated the theory of general relativity", "Albert Einstein"),
               ("formulated the law of universal gravitation", "Isaac Newton"), ("proposed evolution by natural selection", "Charles Darwin"), ("created the World Wide Web", "Tim Berners-Lee"),
               ("developed the movable-type printing press in Europe", "Johannes Gutenberg"), ("created the periodic table of the elements", "Dmitri Mendeleev"), ("proposed the heliocentric model of the solar system", "Nicolaus Copernicus"),
               ("described the three laws of planetary motion", "Johannes Kepler"), ("discovered the four large moons of Jupiter with a telescope", "Galileo Galilei"), ("made the first powered airplane flights", "the Wright brothers"),
               ("developed the first smallpox vaccine", "Edward Jenner"), ("discovered electromagnetic induction", "Michael Faraday"), ("discovered the electron", "J. J. Thomson"), ("conducted pioneering research on radioactivity and won two Nobel Prizes", "Marie Curie")]
UNITS = [("force", "newton"), ("energy", "joule"), ("power", "watt"), ("frequency", "hertz"), ("pressure", "pascal"), ("electric current", "ampere"), ("electrical resistance", "ohm"),
         ("electric charge", "coulomb"), ("thermodynamic temperature", "kelvin"), ("amount of substance", "mole"), ("length", "metre"), ("mass", "kilogram"), ("time", "second"),
         ("electric potential difference", "volt"), ("capacitance", "farad"), ("magnetic flux density", "tesla")]
PLANETS = [("closest to the Sun", "Mercury"), ("second from the Sun", "Venus"), ("known as the Red Planet", "Mars"), ("the largest in the solar system", "Jupiter"), ("famous for its prominent rings", "Saturn"),
           ("the hottest, because of its thick atmosphere", "Venus"), ("the farthest from the Sun", "Neptune"), ("the smallest in the solar system", "Mercury"), ("third from the Sun", "Earth"), ("tilted on its side as it orbits", "Uranus")]
HISTORY = [("World War II ended", 1945), ("World War I began", 1914), ("the French Revolution began", 1789), ("Columbus first reached the Americas", 1492), ("the US Declaration of Independence was adopted", 1776),
           ("the Berlin Wall fell", 1989), ("humans first landed on the Moon", 1969), ("the Titanic sank", 1912), ("the Magna Carta was sealed", 1215), ("Constantinople fell to the Ottomans", 1453),
           ("the Russian Revolution took place", 1917), ("the Great Fire of London occurred", 1666), ("the US Civil War began", 1861), ("the Chernobyl disaster happened", 1986), ("the Wall Street Crash occurred", 1929)]
VITAMINS = [("vitamin C", "scurvy"), ("vitamin D", "rickets"), ("vitamin B1 (thiamine)", "beriberi"), ("vitamin A", "night blindness"), ("iron", "anaemia"), ("iodine", "goitre")]
HTTP = [(200, "OK"), (301, "Moved Permanently"), (401, "Unauthorized"), (403, "Forbidden"), (404, "Not Found"), (500, "Internal Server Error"), (503, "Service Unavailable")]

FACTS = [  # (subject, question, correct, wrong options, explanation)
    ("biology", "Which gas do plants take in for photosynthesis?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Hydrogen"], "Plants absorb carbon dioxide and release oxygen during photosynthesis."),
    ("chemistry", "What is the chemical formula of table salt?", "NaCl", ["KCl", "NaOH", "CaCl2"], "Table salt is sodium chloride, written NaCl."),
    ("chemistry", "What is the chemical formula of water?", "H2O", ["H2O2", "CO2", "O2"], "Water has two hydrogen atoms and one oxygen atom, so its formula is H2O."),
    ("physics", "At standard atmospheric pressure, water boils at what temperature?", "100 degrees Celsius", ["90 degrees Celsius", "110 degrees Celsius", "120 degrees Celsius"], "At sea level pressure pure water boils at 100 degrees Celsius."),
    ("physics", "What is the approximate speed of light in a vacuum?", "300,000 km/s", ["150,000 km/s", "30,000 km/s", "3,000,000 km/s"], "Light travels at about 300,000 kilometres per second in a vacuum."),
    ("biology", "Which organelle is known as the powerhouse of the cell?", "Mitochondria", ["Nucleus", "Ribosome", "Golgi apparatus"], "Mitochondria produce most of the cell's ATP, its energy currency."),
    ("biology", "Which blood cells carry oxygen around the body?", "Red blood cells", ["White blood cells", "Platelets", "Plasma cells"], "Red blood cells contain haemoglobin, which binds oxygen."),
    ("chemistry", "What is the pH of a neutral solution at 25 degrees Celsius?", "7", ["0", "5", "14"], "A pH of 7 is neutral; lower values are acidic and higher values are basic."),
    ("physics", "What is the approximate acceleration due to gravity at the Earth's surface?", "9.8 m/s^2", ["1.6 m/s^2", "3.7 m/s^2", "24.8 m/s^2"], "Near the Earth's surface objects accelerate at about 9.8 metres per second squared."),
    ("physics", "Which force keeps the planets in orbit around the Sun?", "Gravity", ["Friction", "Magnetism", "Air resistance"], "The Sun's gravity pulls the planets toward it and bends their paths into orbits."),
    ("biology", "Which organ produces insulin?", "Pancreas", ["Liver", "Kidney", "Stomach"], "The pancreas makes insulin, which regulates blood sugar."),
    ("earth science", "Which gas makes up most of the Earth's atmosphere?", "Nitrogen", ["Oxygen", "Carbon dioxide", "Argon"], "About 78 percent of the atmosphere is nitrogen."),
    ("physics", "What kind of energy does a moving object have?", "Kinetic energy", ["Potential energy", "Thermal energy", "Chemical energy"], "Energy of motion is called kinetic energy."),
    ("biology", "Which vitamin does the skin produce when exposed to sunlight?", "Vitamin D", ["Vitamin C", "Vitamin B12", "Vitamin K"], "Sunlight on the skin triggers the production of vitamin D."),
    ("biology", "How many chromosomes does a normal human body cell contain?", "46", ["23", "44", "48"], "Humans have 23 pairs of chromosomes, which is 46 in total."),
    ("chemistry", "What is the hardest naturally occurring substance?", "Diamond", ["Quartz", "Steel", "Granite"], "Diamond, a form of carbon, is the hardest natural material."),
    ("biology", "What is the study of fungi called?", "Mycology", ["Botany", "Zoology", "Ecology"], "Mycology is the branch of biology that studies fungi."),
    ("biology", "What is the largest organ of the human body?", "Skin", ["Liver", "Lungs", "Heart"], "The skin has the largest surface area and weight of any organ."),
    ("geography", "Which is the largest ocean on Earth?", "Pacific Ocean", ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean"], "The Pacific covers about a third of the Earth's surface."),
    ("geography", "Which is the highest mountain above sea level?", "Mount Everest", ["K2", "Mount Kilimanjaro", "Mont Blanc"], "Mount Everest rises about 8,849 metres above sea level."),
    ("geography", "Which is the largest continent by area?", "Asia", ["Africa", "North America", "Europe"], "Asia is the largest continent by both area and population."),
    ("geography", "Which is the largest hot desert in the world?", "Sahara", ["Gobi", "Kalahari", "Arabian"], "The Sahara in northern Africa is the largest hot desert."),
    ("geography", "Which country has the largest area in the world?", "Russia", ["Canada", "China", "United States"], "Russia is the largest country by area."),
    ("computer science", "Which data structure follows the last-in first-out principle?", "Stack", ["Queue", "Array", "Linked list"], "In a stack the last element pushed is the first one popped."),
    ("computer science", "Which data structure follows the first-in first-out principle?", "Queue", ["Stack", "Tree", "Hash table"], "In a queue the first element added is the first one removed."),
    ("computer science", "What is the time complexity of binary search on a sorted array of n items?", "O(log n)", ["O(n)", "O(n log n)", "O(1)"], "Binary search halves the search range at each step, so it takes O(log n) time."),
    ("computer science", "What is the average-case lookup time of a well-designed hash table?", "O(1)", ["O(n)", "O(log n)", "O(n^2)"], "A hash table jumps to the right bucket directly, so lookups take constant time on average."),
    ("computer science", "Which sorting algorithm is based on divide and conquer and runs in O(n log n) time in every case?", "Merge sort", ["Bubble sort", "Insertion sort", "Selection sort"], "Merge sort splits the list, sorts the halves and merges them in O(n log n) time."),
    ("computer science", "Which SQL keyword retrieves rows from a table?", "SELECT", ["INSERT", "UPDATE", "DELETE"], "SELECT reads data; the other keywords add, change or remove it."),
    ("computer science", "What does HTML stand for?", "HyperText Markup Language", ["High Transfer Markup Language", "HyperText Machine Language", "Home Tool Markup Language"], "HTML is HyperText Markup Language, used to structure web pages."),
    ("computer science", "What is the default port for HTTPS?", "443", ["80", "21", "22"], "HTTPS uses port 443 by default, while plain HTTP uses port 80."),
    ("economics", "What does GDP stand for?", "Gross Domestic Product", ["Gross Domestic Profit", "General Domestic Product", "Gross Dividend Payment"], "GDP measures the total value of goods and services produced in a country."),
    ("economics", "What is a market with only one seller called?", "Monopoly", ["Oligopoly", "Perfect competition", "Monopsony"], "A monopoly has a single seller and no close substitutes."),
    ("economics", "If demand rises while supply stays the same, what usually happens to the price?", "It rises", ["It falls", "It stays the same", "It becomes zero"], "More buyers chasing the same supply push the price up."),
    ("economics", "What is opportunity cost?", "The value of the next best alternative given up", ["The price printed on a product", "The cost of producing one more unit", "The interest paid on a loan"], "Opportunity cost is what you give up when you choose one option over another."),
    ("economics", "What is inflation?", "A general rise in prices over time", ["A fall in unemployment", "A rise in a currency's exchange rate", "An increase in savings"], "Inflation means money buys less because prices rise on average."),
    ("astronomy", "What is the name of the galaxy that contains our solar system?", "The Milky Way", ["Andromeda", "Triangulum", "Whirlpool"], "Our solar system is located in the Milky Way galaxy."),
    ("astronomy", "What causes the seasons on Earth?", "The tilt of the Earth's axis", ["The distance from the Sun", "The Moon's gravity", "Solar flares"], "The tilted axis changes how directly sunlight hits each hemisphere during the year."),
    ("biology", "Which process do cells use to divide into two identical daughter cells?", "Mitosis", ["Meiosis", "Osmosis", "Diffusion"], "Mitosis produces two genetically identical cells."),
    ("chemistry", "What type of bond involves the sharing of electron pairs between atoms?", "Covalent bond", ["Ionic bond", "Hydrogen bond", "Metallic bond"], "In a covalent bond atoms share pairs of electrons."),
]

FACTS += [
    ("biology", "Which body system includes the heart and blood vessels?", "Circulatory system", ["Respiratory system", "Digestive system", "Nervous system"], "The heart pumps blood through vessels, which together form the circulatory system."),
    ("biology", "Which body system includes the lungs?", "Respiratory system", ["Circulatory system", "Skeletal system", "Endocrine system"], "The lungs bring oxygen in and remove carbon dioxide as part of the respiratory system."),
    ("biology", "How many chambers does a human heart have?", "4", ["2", "3", "5"], "The human heart has two atria and two ventricles, so four chambers."),
    ("biology", "Which cell organelle makes proteins?", "Ribosome", ["Mitochondrion", "Lysosome", "Vacuole"], "Ribosomes assemble amino acids into proteins."),
    ("biology", "Which structure is found in plant cells but not in animal cells?", "Cell wall", ["Nucleus", "Cell membrane", "Mitochondrion"], "Plant cells have a rigid cell wall outside the membrane; animal cells do not."),
    ("biology", "Which organelle carries out photosynthesis?", "Chloroplast", ["Mitochondrion", "Nucleus", "Ribosome"], "Chloroplasts contain chlorophyll and capture light energy for photosynthesis."),
    ("biology", "Which is the longest bone in the human body?", "Femur", ["Humerus", "Tibia", "Spine"], "The femur, or thigh bone, is the longest and strongest bone."),
    ("geology", "Which type of rock forms when magma or lava cools and hardens?", "Igneous rock", ["Sedimentary rock", "Metamorphic rock", "Limestone"], "Igneous rocks form from cooled molten rock."),
    ("geology", "Which type of rock forms from layers of compacted sediment?", "Sedimentary rock", ["Igneous rock", "Metamorphic rock", "Volcanic rock"], "Sedimentary rocks form as sediments pile up and are pressed into layers."),
    ("geology", "Which type of rock is changed by heat and pressure?", "Metamorphic rock", ["Igneous rock", "Sedimentary rock", "Sandstone"], "Heat and pressure transform existing rock into metamorphic rock."),
    ("meteorology", "Which instrument measures atmospheric pressure?", "Barometer", ["Thermometer", "Anemometer", "Hygrometer"], "A barometer measures air pressure."),
    ("meteorology", "Which instrument measures wind speed?", "Anemometer", ["Barometer", "Seismograph", "Thermometer"], "An anemometer measures how fast the wind blows."),
    ("geology", "Which instrument records earthquakes?", "Seismograph", ["Barometer", "Anemometer", "Telescope"], "A seismograph records ground motion from earthquakes."),
    ("mathematics", "How many degrees do the angles of a triangle add up to?", "180", ["90", "270", "360"], "The interior angles of any triangle sum to 180 degrees."),
    ("mathematics", "How many sides does a hexagon have?", "6", ["5", "7", "8"], "A hexagon is a polygon with six sides."),
    ("mathematics", "What is the value of pi rounded to two decimal places?", "3.14", ["3.41", "3.12", "2.71"], "Pi is the ratio of a circle's circumference to its diameter, about 3.14."),
    ("mathematics", "What is the smallest prime number?", "2", ["0", "1", "3"], "Two is the smallest prime and the only even prime."),
    ("mathematics", "What is the result of any number multiplied by zero?", "0", ["1", "The number itself", "Undefined"], "Multiplying any number by zero gives zero."),
    ("computer science", "Which keyword defines a function in Python?", "def", ["func", "function", "lambda"], "Python functions are defined with the def keyword."),
    ("computer science", "What does CSS stand for?", "Cascading Style Sheets", ["Computer Style Sheets", "Creative Style System", "Cascading Script Syntax"], "CSS describes how HTML elements look on a page."),
    ("computer science", "Which symbol starts a single-line comment in Python?", "#", ["//", "--", "/*"], "In Python everything after # on a line is a comment."),
    ("computer science", "What does API stand for?", "Application Programming Interface", ["Applied Program Integration", "Automated Process Interface", "Application Protocol Index"], "An API defines how software components talk to each other."),
    ("computer science", "What does RAM stand for?", "Random Access Memory", ["Read Access Memory", "Rapid Active Memory", "Random Allocation Module"], "RAM is the computer's fast working memory."),
    ("computer science", "Which Unix command lists the files in a directory?", "ls", ["cd", "rm", "cat"], "ls lists directory contents."),
    ("computer science", "In Python 3 what type does 3 / 2 produce?", "A float (1.5)", ["An integer (1)", "A string", "A boolean"], "The / operator always returns a float in Python 3."),
    ("economics", "In which market structure are there many sellers of identical products and no single seller sets the price?", "Perfect competition", ["Monopoly", "Oligopoly", "Cartel"], "Perfect competition has many sellers and price-taking firms."),
    ("economics", "What do we call a tax that takes a larger percentage from higher incomes?", "Progressive tax", ["Flat tax", "Regressive tax", "Sales tax"], "A progressive tax rate rises with income."),
    ("economics", "What is the term for a sustained fall in the general price level?", "Deflation", ["Inflation", "Stagflation", "Recession"], "Deflation means prices fall across the economy over time."),
    ("government", "In many democracies which branch of government makes the laws?", "The legislature", ["The judiciary", "The executive", "The military"], "The legislature writes and passes laws; the executive enforces them and the judiciary interprets them."),
    ("history", "Which ancient civilisation built the pyramids of Giza?", "Ancient Egypt", ["Ancient Greece", "The Roman Empire", "The Maya"], "The Giza pyramids were built by the ancient Egyptians as royal tombs."),
    ("history", "Which empire was ruled from Rome in antiquity?", "The Roman Empire", ["The Ottoman Empire", "The Persian Empire", "The Byzantine Empire"], "The Roman Empire was governed from Rome in antiquity."),
    ("physics", "What do we call the bending of light as it passes from air into water?", "Refraction", ["Reflection", "Diffraction", "Absorption"], "Light changes direction when it changes speed in a new medium, which is refraction."),
    ("physics", "Which type of wave is sound?", "Longitudinal wave", ["Transverse wave", "Electromagnetic wave", "Standing light wave"], "Sound is a pressure wave whose particles move along the direction of travel."),
    ("physics", "What is the unit of electrical resistance?", "Ohm", ["Volt", "Ampere", "Watt"], "Resistance is measured in ohms."),
    ("chemistry", "What is the most abundant element in the universe?", "Hydrogen", ["Helium", "Oxygen", "Carbon"], "Hydrogen makes up roughly three quarters of ordinary matter in the universe."),
    ("chemistry", "Which gas is produced when an acid reacts with a metal such as zinc?", "Hydrogen", ["Oxygen", "Carbon dioxide", "Nitrogen"], "Acids react with reactive metals to give a salt and hydrogen gas."),
    ("chemistry", "What is the chemical symbol for sodium?", "Na", ["So", "Sd", "S"], "Sodium's symbol Na comes from its Latin name natrium."),
    ("astronomy", "What is the closest star to the Earth?", "The Sun", ["Proxima Centauri", "Sirius", "Polaris"], "The Sun is the nearest star, about 150 million kilometres away."),
    ("astronomy", "What is a light-year a unit of?", "Distance", ["Time", "Speed", "Brightness"], "A light-year is the distance light travels in one year."),
]


def num(v) -> str:
    return f"{v:,}".replace(",", "") if isinstance(v, int) else f"{v:g}"


def near(r, x, k=3, lo=None):
    out = set()
    while len(out) < k:
        d = r.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]) if abs(x) > 12 else r.choice([-3, -2, -1, 1, 2, 3, 4])
        v = x + d
        if v != x and (lo is None or v >= lo):
            out.add(v)
    return list(out)


# --- computed arithmetic ------------------------------------------------------------------------------------------------------------
def q_mul(r):
    a, b = r.randrange(6, 30), r.randrange(3, 20)
    return "mathematics", f"What is {a} x {b}?", a * b, near(r, a * b), f"{a} x {b} = {a * b}."


def q_percent(r):
    p, n = r.choice([10, 15, 20, 25, 30, 40, 50, 75]), r.randrange(2, 40) * 20
    ans = n * p / 100
    return "mathematics", f"What is {p}% of {n}?", ans, [x for x in {ans + d for d in (-p, p, n / 10, -n / 10)} if x != ans and x > 0][:3] or near(r, int(ans)), f"{p}% of {n} is {n} x {p}/100 = {num(ans)}."


def q_prime(r):
    primes = [p for p in range(10, 100) if all(p % d for d in range(2, int(p ** .5) + 1))]
    comp = [c for c in range(10, 100) if c not in primes and c % 2 and c % 3]
    ans = r.choice(primes)
    return "mathematics", "Which of the following numbers is prime?", ans, r.sample(comp, 3), f"{ans} has no divisors other than 1 and itself; the other options have smaller factors."


def q_gcd(r):
    g = r.choice([2, 3, 4, 5, 6, 7, 8, 9, 12])
    a, b = g * r.choice([2, 3, 5, 7]), g * r.choice([4, 6, 9, 11])
    ans = math.gcd(a, b)
    return "mathematics", f"What is the greatest common divisor of {a} and {b}?", ans, [x for x in [1, 2, 3, 4, 6, 8, 12, ans * 2] if x != ans][:3], f"The largest number dividing both {a} and {b} is {ans}."


def q_lcm(r):
    a, b = r.randrange(3, 13), r.randrange(3, 13)
    ans = math.lcm(a, b)
    return "mathematics", f"What is the least common multiple of {a} and {b}?", ans, list({a * b + 0 if a * b != ans else ans * 2, ans + a, ans - b if ans - b > 0 else ans + b + 1}), f"The smallest number divisible by both {a} and {b} is {ans}."


def q_linear(r):
    x, a, b = r.randrange(2, 15), r.randrange(2, 9), r.randrange(1, 20)
    return "algebra", f"Solve for x: {a}x + {b} = {a * x + b}.", x, near(r, x, lo=0), f"Subtract {b} from both sides to get {a}x = {a * x}, so x = {x}."


def q_mean(r):
    xs = [r.randrange(2, 30) for _ in range(5)]
    xs[-1] += (-sum(xs)) % 5
    ans = sum(xs) // 5
    return "statistics", f"What is the mean of {', '.join(map(str, xs))}?", ans, near(r, ans), f"The sum is {sum(xs)} and dividing by 5 gives {ans}."


def q_area(r):
    w, h = r.randrange(3, 20), r.randrange(3, 20)
    return "geometry", f"What is the area of a rectangle with width {w} and height {h}?", w * h, [2 * (w + h), w * h + w, w + h], f"Area = width x height = {w} x {h} = {w * h}."


def q_binary(r):
    n = r.randrange(5, 60)
    return "computer science", f"What is the binary number {format(n, 'b')} in decimal?", n, near(r, n, lo=1), f"{format(n, 'b')} in binary equals {n} in decimal."


def q_square(r):
    n = r.randrange(6, 26)
    return "mathematics", f"What is the square root of {n * n}?", n, near(r, n, lo=1), f"{n} x {n} = {n * n}, so the square root is {n}."


def q_fraction(r):
    a, b = r.choice([(1, 2), (1, 3), (1, 4), (3, 4), (2, 5)]), r.choice([(1, 4), (1, 5), (1, 6), (1, 2)])
    from fractions import Fraction
    s = Fraction(*a) + Fraction(*b)
    opts = {f"{s.numerator}/{s.denominator}"}
    wrong = {f"{a[0] + b[0]}/{a[1] + b[1]}", f"{a[0] * b[0]}/{a[1] * b[1]}", f"{a[0] + b[0]}/{max(a[1], b[1])}", f"{s.numerator + 1}/{s.denominator}"} - opts
    return "mathematics", f"What is {a[0]}/{a[1]} + {b[0]}/{b[1]}?", f"{s.numerator}/{s.denominator}", list(wrong)[:3], f"With a common denominator the sum is {s.numerator}/{s.denominator}."


COMPUTED = [q_mul, q_percent, q_prime, q_gcd, q_lcm, q_linear, q_mean, q_area, q_binary, q_square, q_fraction]

def q_unit(r):
    k, m = r.choice([("kilometres", "metres", 1000), ("hours", "minutes", 60), ("minutes", "seconds", 60), ("metres", "centimetres", 100), ("kilograms", "grams", 1000)])[:2], None
    big, small, f = r.choice([("kilometres", "metres", 1000), ("hours", "minutes", 60), ("minutes", "seconds", 60), ("metres", "centimetres", 100), ("kilograms", "grams", 1000)])
    n = r.randrange(2, 30)
    return "mathematics", f"How many {small} are in {n} {big}?", n * f, [n * f * 10, n * f // 10 if n * f // 10 else n + 1, n + f], f"One {big[:-1] if big.endswith('s') else big} is {f} {small}, so {n} x {f} = {n * f}."


def q_speed(r):
    v, t = r.choice([40, 50, 60, 80, 90]), r.choice([2, 3, 4, 5])
    return "physics", f"A car travels at a constant {v} km/h for {t} hours. How far does it go?", v * t, [v + t, v * t + v, v * (t + 1)], f"Distance = speed x time = {v} x {t} = {v * t} km."


def q_ohm(r):
    i, rr = r.randrange(2, 9), r.choice([5, 10, 20, 50])
    return "physics", f"What voltage is needed to drive a current of {i} A through a resistance of {rr} ohms?", i * rr, [i + rr, rr // i if rr // i else 1, i * rr * 2], f"Ohm's law gives V = I x R = {i} x {rr} = {i * rr} volts."


def q_density(r):
    m, v = r.choice([(100, 20), (60, 12), (90, 30), (48, 8), (150, 50)])
    return "physics", f"An object has a mass of {m} g and a volume of {v} cm^3. What is its density in g/cm^3?", m // v, [m // v + 1, m // v + 2, v], f"Density = mass / volume = {m} / {v} = {m // v}."


def q_sequence(r):
    a, d, n = r.randrange(1, 10), r.randrange(2, 8), r.randrange(5, 12)
    return "mathematics", f"What is the {n}th term of the arithmetic sequence that starts with {a} and increases by {d} each time?", a + (n - 1) * d, [a + n * d, a + (n - 2) * d, a * n], f"The nth term is a + (n - 1) x d = {a} + {n - 1} x {d} = {a + (n - 1) * d}."


def q_order_ops(r):
    a, b, c = r.randrange(2, 9), r.randrange(2, 9), r.randrange(2, 9)
    return "mathematics", f"What is {a} + {b} x {c}?", a + b * c, [(a + b) * c, a * b + c, a + b + c], f"Multiplication comes first: {b} x {c} = {b * c}, then add {a} to get {a + b * c}."


def q_power(r):
    b, e = r.randrange(2, 7), r.randrange(2, 6)
    return "mathematics", f"What is {b} to the power of {e}?", b ** e, [b * e, b ** (e - 1), b ** (e + 1)], f"{b}^{e} means {b} multiplied by itself {e} times, which is {b ** e}."


def q_simple_interest(r):
    p, rate, t = r.randrange(1, 10) * 100, r.choice([2, 4, 5, 10]), r.randrange(2, 6)
    return "mathematics", f"What is the simple interest on {p} euros at {rate}% per year for {t} years?", p * rate * t // 100, [p * rate * t // 10, p * rate // 100, p * (rate + t)], f"Simple interest = P x r x t = {p} x {rate}/100 x {t} = {p * rate * t // 100}."


def q_mode(r):
    base = r.sample(range(1, 20), 4)
    m = r.choice(base)
    xs = base + [m, m]
    r.shuffle(xs)
    return "statistics", f"What is the mode of {', '.join(map(str, xs))}?", m, [x for x in base if x != m][:3], f"{m} appears three times, more than any other value."


def q_prob(r):
    n = r.choice([2, 3, 4, 5, 6, 8, 10])
    return "probability", f"A fair die with {n} equal sides is rolled once. What is the probability of getting a specific number?", f"1/{n}", [f"1/{n + 1}", f"2/{n}", f"1/{n - 1}" if n > 2 else "1/1"], f"All {n} outcomes are equally likely, so one specific outcome has probability 1/{n}."


COMPUTED += [q_unit, q_speed, q_ohm, q_density, q_sequence, q_order_ops, q_power, q_simple_interest, q_mode, q_prob]

# --- table-driven questions ---------------------------------------------------------------------------------------------------------
TABLES = []   # each: dict(subject, rows=[(key, value)], fwd=[templates with {k}], rev=[templates with {v}], fact='{k}: {v}' )


def table(subject, rows, fwd, rev, fact_f, fact_r=None):
    TABLES.append(dict(subject=subject, rows=rows, fwd=fwd, rev=rev, fact=fact_f, fact_r=fact_r or fact_f))


table("geography", CAPITALS, ["What is the capital of {k}?", "Which city is the capital of {k}?"], ["{v} is the capital of which country?", "Which country has {v} as its capital?"], "{v} is the capital of {k}.")
table("chemistry", [(n, s) for n, s, _ in ELEMENTS], ["What is the chemical symbol for {k}?", "Which symbol represents the element {k}?"], ["Which element has the chemical symbol {v}?", "The symbol {v} stands for which element?"], "The symbol for {k} is {v}.")
table("chemistry", [(n, z) for n, _, z in ELEMENTS], ["What is the atomic number of {k}?", "How many protons does an atom of {k} have?"], ["Which element has atomic number {v}?", "Which element has {v} protons in its nucleus?"], "{k} has atomic number {v}.")
table("literature", AUTHORS, ["Who wrote {k}?", "Who is the author of {k}?"], ["Which of the following was written by {v}?", "Which work is by {v}?"], "{k} was written by {v}.")
table("history of science", DISCOVERIES, ["Who {k}?", "Which person or group {k}?"], [], "{v} {k}.")
table("physics", UNITS, ["What is the SI unit of {k}?", "In the International System of Units, which unit measures {k}?"], ["The {v} is the SI unit of which quantity?", "Which quantity is measured in {v}s?"], "The SI unit of {k} is the {v}.")
table("astronomy", PLANETS, ["Which planet is {k}?", "Which planet is described as {k}?"], [], "{v} is {k}.")
table("history", [(e, y) for e, y in HISTORY], ["In which year {k}?", "When did it happen that {k}? Pick the year."], ["Which event happened in {v}?"], "{k} in {v}.")
table("nutrition and health", VITAMINS, ["A deficiency of {k} causes which condition?", "Lack of {k} in the diet leads to which disease?"], ["Which nutrient deficiency causes {v}?"], "A lack of {k} causes {v}.")
table("computer science", [(str(c), n) for c, n in HTTP], ["What does HTTP status code {k} mean?", "Which message goes with HTTP status {k}?"], ["Which HTTP status code means \"{v}\"?"], "HTTP {k} means {v}.")

LANDMARKS = [("the Eiffel Tower", "France"), ("the Colosseum", "Italy"), ("the Sagrada Familia", "Spain"), ("the Acropolis", "Greece"), ("Big Ben", "United Kingdom"), ("Machu Picchu", "Peru"),
             ("the Taj Mahal", "India"), ("Christ the Redeemer", "Brazil"), ("the Pyramids of Giza", "Egypt"), ("the Statue of Liberty", "United States"), ("Petra", "Jordan"), ("Angkor Wat", "Cambodia"),
             ("Mount Fuji", "Japan"), ("the Brandenburg Gate", "Germany"), ("the Charles Bridge", "Czech Republic"), ("the Blue Mosque", "Turkey")]
LANGS = [("Brazil", "Portuguese"), ("Egypt", "Arabic"), ("Japan", "Japanese"), ("Austria", "German"), ("Argentina", "Spanish"), ("Netherlands", "Dutch"), ("Poland", "Polish"), ("Turkey", "Turkish"),
         ("Thailand", "Thai"), ("Hungary", "Hungarian"), ("Greece", "Greek"), ("China", "Mandarin Chinese"), ("Russia", "Russian"), ("Italy", "Italian")]
MONEY = [("Japan", "yen"), ("the United Kingdom", "pound sterling"), ("India", "rupee"), ("Switzerland", "Swiss franc"), ("Poland", "zloty"), ("Sweden", "krona"), ("Brazil", "real"), ("Mexico", "peso"),
         ("South Korea", "won"), ("Russia", "rouble"), ("Turkey", "lira"), ("Denmark", "krone"), ("Norway", "krone"), ("Czech Republic", "koruna")]
ART = [("The Starry Night", "Vincent van Gogh"), ("Guernica", "Pablo Picasso"), ("The Persistence of Memory", "Salvador Dali"), ("The ceiling of the Sistine Chapel", "Michelangelo"), ("The Scream", "Edvard Munch"),
       ("Mona Lisa", "Leonardo da Vinci"), ("Girl with a Pearl Earring", "Johannes Vermeer"), ("The Birth of Venus", "Sandro Botticelli"), ("Water Lilies", "Claude Monet"), ("The Night Watch", "Rembrandt")]
MUSIC = [("The Four Seasons", "Antonio Vivaldi"), ("The Magic Flute", "Wolfgang Amadeus Mozart"), ("Swan Lake", "Pyotr Tchaikovsky"), ("Bolero", "Maurice Ravel"), ("Symphony No. 9 with the Ode to Joy", "Ludwig van Beethoven"),
         ("The Brandenburg Concertos", "Johann Sebastian Bach"), ("The Blue Danube waltz", "Johann Strauss II"), ("Carmen", "Georges Bizet"), ("The Planets", "Gustav Holst"), ("The Rite of Spring", "Igor Stravinsky")]
PHIL = [("Critique of Pure Reason", "Immanuel Kant"), ("The Republic", "Plato"), ("Nicomachean Ethics", "Aristotle"), ("Leviathan", "Thomas Hobbes"), ("Thus Spoke Zarathustra", "Friedrich Nietzsche"),
        ("Meditations on First Philosophy", "Rene Descartes"), ("The Wealth of Nations", "Adam Smith"), ("The Social Contract", "Jean-Jacques Rousseau")]
table("geography", LANDMARKS, ["In which country is {k}?", "Which country is home to {k}?"], ["Which of these landmarks is in {v}?"], "{k} is in {v}.")
table("geography", LANGS, ["What is the main official language of {k}?", "Which language is mainly spoken in {k}?"], ["In which country is {v} the main language?"], "The main language of {k} is {v}.")
table("economics", MONEY, ["What is the currency of {k}?", "Which currency is used in {k}?"], ["Which country uses the {v}?"], "The currency of {k} is the {v}.")
table("art", ART, ["Who painted {k}?", "Which artist created {k}?"], ["Which of these works was created by {v}?"], "{k} was created by {v}.")
table("music", MUSIC, ["Who composed {k}?", "Which composer wrote {k}?"], ["Which of these works was composed by {v}?"], "{k} was composed by {v}.")
table("philosophy and economics", PHIL, ["Who wrote {k}?", "Who is the author of {k}?"], ["Which of these works was written by {v}?"], "{k} was written by {v}.")


def build_table_q(r):
    t = r.choice(TABLES)
    rows = t["rows"]
    can_rev = bool(t["rev"])
    rev = can_rev and r.random() < 0.45
    k, v = r.choice(rows)
    if rev:
        q = r.choice(t["rev"]).format(v=v)
        correct, key_of = k, {kk: vv for kk, vv in rows}
        pool = [x for x, _ in rows if x != k and key_of[x] != v]
        fact = lambda opt: t["fact"].format(k=opt, v=key_of[opt])
    else:
        q = r.choice(t["fwd"]).format(k=k)
        correct, val_of = v, {kk: vv for kk, vv in rows}
        pool = list({vv for _, vv in rows if vv != v})
        inv = {}
        for kk, vv in rows:
            inv.setdefault(vv, kk)
        fact = lambda opt: t["fact"].format(k=inv[opt], v=opt) if opt in inv else str(opt)
    wrong = r.sample(pool, 3)
    return t["subject"], q, correct, wrong, fact(correct), {opt: fact(opt) for opt in wrong}


def build_fact_q(r):
    s, q, c, w, e = r.choice(FACTS)
    return s, q, c, w, e, {}


def make_mcq(kind: str, r: random.Random):
    if kind == "computed":
        s, q, c, w, e = r.choice(COMPUTED)(r)
        w = [x for x in dict.fromkeys(w) if x != c][:3]
        if len(w) < 3:
            return None
        return s, q, c, w, e, {}
    if kind == "table":
        return build_table_q(r)
    return build_fact_q(r)


LAYOUTS = [
    "The following is a multiple choice question about {subject}.\n\n{q}\n{opts}\nAnswer:",
    "{q}\n\n{opts}\n\nWhich option is correct?",
    "Question: {q}\nOptions:\n{opts}\nPick the correct answer.",
    "Answer this {subject} question.\n\n{q}\n{opts}",
    "{q}\n{opts}\nGive the letter of the correct option.",
]
OPT_FMT = [lambda L, t: f"{L}. {t}", lambda L, t: f"({L}) {t}", lambda L, t: f"{L}) {t}"]


def render(row, r: random.Random, force_style=None):
    subject, q, correct, wrong, why, why_wrong = row
    options = [correct] + list(wrong)
    r.shuffle(options)
    letter = LET[options.index(correct)]
    fmt = r.choice(OPT_FMT)
    opts = "\n".join(fmt(LET[i], num(o) if isinstance(o, (int, float)) else str(o)) for i, o in enumerate(options))
    prompt = r.choice(LAYOUTS).format(subject=subject, q=q, opts=opts)
    style = force_style or r.choices(["direct", "answer_is", "cot"], [0.25, 0.15, 0.60])[0]
    ctext = num(correct) if isinstance(correct, (int, float)) else str(correct)
    if style == "direct":
        ans = letter
    elif style == "answer_is":
        ans = f"The answer is {letter}. {ctext}."
    else:
        ans = r.choice(["Let me check the options.", "I'll recall the relevant fact first.", "Let me think about what the question asks."]) + f"\n\n{why}\n\n" + \
            r.choice([f"Only option {letter} matches this.", f"That corresponds to option {letter}."]) + f"\n\nFinal answer: {letter}. {ctext}"
    return prompt, ans, letter, options, style


def explain_wrong(options, letter, why_wrong, correct, why):
    lines = []
    for i, o in enumerate(options):
        if LET[i] == letter:
            continue
        key = o
        if key in why_wrong:
            lines.append(f"{LET[i]} is wrong because {why_wrong[key]}")
        else:
            lines.append(f"{LET[i]} ({num(o) if isinstance(o, (int, float)) else o}) does not fit the question.")
    return f"{letter} is correct: {why[0].lower() + why[1:]}\n\n" + "\n".join(lines)


def build_one(seed: int, kind: str) -> dict | None:
    r = random.Random(f"mcq:{kind}:{seed}")
    row = make_mcq(kind, r)
    if row is None:
        return None
    two = kind in ("table", "computed") and row[5] != {} and r.random() < 0.40
    prompt, ans, letter, options, style = render(row, r, force_style=("cot" if False else None))
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
    tags = ["mcq", kind, style]
    if two:
        msgs += [{"role": "user", "content": r.choice(["Why are the other options wrong?", "Can you explain why the other choices don't work?", "Briefly explain the other options."])},
                 {"role": "assistant", "content": explain_wrong(options, letter, row[5], row[2], row[4])}]
        tags.append("two_turn")
    if r.random() < 0.12:
        msgs.insert(0, {"role": "system", "content": r.choice(["Answer multiple choice questions accurately.", "You are a knowledgeable assistant. Pick the best option."])})
    return {"id": f"mcq/{kind}/{seed}", "messages": msgs, "tools": [],
            "meta": {"family": f"mcq_{row[0].replace(' ', '_')}", "category": "knowledge", "difficulty": "easy", "kind": "mcq", "seed": seed, "tags": tags,
                     "n_tool_calls": 0, "n_turns": sum(m["role"] == "user" for m in msgs), "n_tools_offered": 0, "answer": letter}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--out", default="data/raw/mcq_knowledge.jsonl")
    ap.add_argument("--seed-offset", type=int, default=30_000)
    a = ap.parse_args()
    mix = {"table": 0.50, "computed": 0.30, "fact": 0.20}
    rows, seen = [], set()
    for kind, w in mix.items():
        want, k, made = int(a.n * w), 0, 0
        while made < want and k < want * 6:
            rec = build_one(a.seed_offset + k, kind)
            k += 1
            if rec is None:
                continue
            key = rec["messages"][-2 if "two_turn" not in rec["meta"]["tags"] else -4]["content"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
            made += 1
    random.Random(3).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    from collections import Counter
    print(f"{len(rows)} questions -> {a.out}; kinds {dict(Counter(x['meta']['tags'][1] for x in rows))}; two_turn {sum('two_turn' in x['meta']['tags'] for x in rows)}; letters {dict(Counter(x['meta']['answer'] for x in rows))}")


if __name__ == "__main__":
    main()
