#include <ctype.h>
#include <stdio.h>
#include <string.h>

enum TokenKind { TOK_NUMBER, TOK_PLUS, TOK_STAR, TOK_LPAREN, TOK_RPAREN, TOK_END };

struct Token {
    enum TokenKind kind;
    int value;
};

static const char *cursor;

static struct Token next_token(void) {
    while (*cursor && isspace((unsigned char)*cursor)) {
        cursor++;
    }
    struct Token tok = {TOK_END, 0};
    if (*cursor == '\0') {
        return tok;
    }
    if (isdigit((unsigned char)*cursor)) {
        int value = 0;
        while (isdigit((unsigned char)*cursor)) {
            value = value * 10 + (*cursor - '0');
            cursor++;
        }
        tok.kind = TOK_NUMBER;
        tok.value = value;
        return tok;
    }
    switch (*cursor++) {
    case '+':
        tok.kind = TOK_PLUS;
        break;
    case '*':
        tok.kind = TOK_STAR;
        break;
    default:
        tok.kind = TOK_END;
        break;
    }
    return tok;
}

int main(void) {
    cursor = "12 + 3 * 4";
    struct Token tok;
    int sum = 0;
    while ((tok = next_token()).kind != TOK_END) {
        if (tok.kind == TOK_NUMBER) {
            sum += tok.value;
        }
    }
    printf("sum of literals = %d\n", sum);
    return 0;
}
