import numpy as np
import random
import matplotlib.pyplot as mp


class Element:

    def __init__(self, alias):
        self.alias = alias
        self.count = 0
        self.next_rock_count = 0
        self.next_paper_count = 0
        self.next_scissors_count = 0
        self.next_probabilities = None

    def p_next_rock(self):
        return self.next_rock_count / self.count

    def p_next_paper(self):
        return self.next_paper_count / self.count

    def p_next_scissors(self):
        return self.next_scissors_count / self.count

    def all_p_next(self):
        return [self.p_next_rock(), self.p_next_paper(), self.p_next_scissors()]

    def can_calculate_probabilities(self):
        return self.count != 0 and (
                self.next_rock_count != 0 or self.next_paper_count != 0 or self.next_scissors_count != 0)

    def increment_next_count(self, next_element):
        self.count += 1
        next_element_alias = next_element.alias
        if next_element_alias == 'r':
            self.next_rock_count += 1
        elif next_element_alias == 'p':
            self.next_paper_count += 1
        elif next_element_alias == 's':
            self.next_scissors_count += 1
        if self.can_calculate_probabilities():
            self.next_probabilities = self.all_p_next()

    def __str__(self):
        return f'alias = {self.alias}, ' \
               f'count = {self.count}, ' \
               f'next_rock_count = {self.next_rock_count}, ' \
               f'next_paper_count = {self.next_paper_count}, ' \
               f'next_scissors_count = {self.next_scissors_count}, ' \
               f'next_probabilities = {self.next_probabilities}'

    def __repr__(self):
        return self.__str__()


class Register:

    def __init__(self):
        self.games = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.cash = 0
        self.cash_log = []

    def update(self, round_result):
        self.games += 1
        if round_result == 'win':
            self.wins += 1
            self.cash += 1
        elif round_result == 'loss':
            self.losses += 1
            self.cash -= 1
        else:
            self.ties += 1
        self.cash_log.append(int(self.cash))

    def __str__(self):
        return f'games = {self.games}, ' \
               f'wins = {self.wins}, ' \
               f'losses = {self.losses}, ' \
               f'ties = {self.ties}, ' \
               f'cash = {self.cash}'

    def __repr__(self):
        return self.__str__()


def guess_next_player_element(last_player_element, elements):
    if last_player_element.next_probabilities is not None:
        probabilities = last_player_element.next_probabilities
        return np.random.choice(elements, p=probabilities)
    else:
        return random.choice(elements)


def find_by_alias(alias, elements):
    for element in elements:
        if element.alias == alias:
            return element
    return None


def get_counter_element(element, elements):
    counter_elements_aliases = {'r': 'p', 'p': 's', 's': 'r'}
    counter_element_alias = counter_elements_aliases.get(element.alias)
    return find_by_alias(counter_element_alias, elements)


def show_graph(register):
    cash_state = register.cash_log
    game_number = [i for i in range(1, len(cash_state) + 1)]
    mp.plot(game_number, cash_state)
    mp.xlabel('Game number')
    mp.ylabel('Cash')
    mp.title('Cash register changes')
    mp.show()


def handle_player_input(elements, register):
    while True:
        player_input = str(
            input('Enter \'r\' - rock, \'p\' - paper, \'s\' - scissors, \'graph\' to show graph or \'quit\' to quit\n'))
        if player_input == 'quit':
            quit()
        if player_input == 'graph':
            show_graph(register)
        element = find_by_alias(player_input, elements)
        if element is not None:
            return element


def evaluate_round(our_element, player_element, elements):
    our_element_counter = get_counter_element(our_element, elements)
    if our_element_counter is player_element:
        return 'win'
    player_element_counter = get_counter_element(player_element, elements)
    if player_element_counter is our_element:
        return 'loss'
    return 'tie'


def print_state(elements):
    for element in elements:
        print(element)
    print('-----------------------------------------------------------------------------------------------------------')


def print_round_result(computer_element, round_result, register):
    print(round_result.upper(), 'computer played', computer_element.alias, '(', register, ')\n')


def play_first_round(elements, register):
    computer_element = random.choice(elements)

    player_element = handle_player_input(elements, register)

    round_result = evaluate_round(computer_element, player_element, elements)
    register.update(round_result)

    print_round_result(computer_element, round_result, register)
    print_state(elements)

    return player_element


def play_round(last_player_element, elements, register):
    next_player_element_prediction = guess_next_player_element(last_player_element, elements)
    computer_element = get_counter_element(next_player_element_prediction, elements)

    player_element = handle_player_input(elements, register)

    last_player_element.increment_next_count(player_element)

    round_result = evaluate_round(computer_element, player_element, elements)
    register.update(round_result)

    print_round_result(computer_element, round_result, register)
    print_state(elements)

    return player_element


def run_game():
    rock = Element('r')
    paper = Element('p')
    scissors = Element('s')

    elements = [rock, paper, scissors]
    register = Register()

    print_state(elements)

    last_player_element = play_first_round(elements, register)
    while True:
        last_player_element = play_round(last_player_element, elements, register)


run_game()
