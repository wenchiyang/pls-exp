action(0)::action(stay);
action(1)::action(up);
action(2)::action(down);
action(3)::action(left);
action(4)::action(right).


ghost(0)::ghost(0,  0,  1).
ghost(1)::ghost(0,  0, -1).
ghost(2)::ghost(0, -1,  0).
ghost(3)::ghost(0,  1,  0).

ghost(4)::ghost(0,  0,  2).
ghost(5)::ghost(0,  0, -2).
ghost(6)::ghost(0, -2,  0).
ghost(7)::ghost(0,  2,  0).

ghost(8)::ghost(0,  1,  1).
ghost(9)::ghost(0,  1, -1).
ghost(10)::ghost(0, -1,  1).
ghost(11)::ghost(0, -1, -1).


% transition(Action, NextPos)
transition(X, Y, stay, X, Y).
transition(X, Y, left, X1, Y) :- X1 is X - 1.
transition(X, Y, right, X1, Y) :- X1 is X + 1.
transition(X, Y, up, X, Y1):- Y1 is Y + 1.
transition(X, Y, down, X, Y1):- Y1 is Y - 1.

ghost(T1, X1, Y1) :-
    T is T1 - 1,
    T >= 0,
    ghost(T, X, Y),
    transition(X, Y, _, X1, Y1).


agent(1, X1, Y1) :-
    action(A),
    transition(0, 0, A, X1, Y1).

agent(T1, X, Y):-
    T1 > 1,
    T is T1 - 1,
    agent(T, X, Y).

unsafe_next :- ghost(1, X1, Y1), agent(1, X1, Y1).
unsafe_next :- ghost(2, X1, Y1), agent(2, X1, Y1).
safe_next :- \+ unsafe_next.

safe_action(A):- action(A).
