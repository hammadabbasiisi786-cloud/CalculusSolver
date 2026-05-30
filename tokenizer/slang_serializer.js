const OPEN = "STRUCT:OPEN";
const CLOSE = "STRUCT:CLOSE";
const SEP = "STRUCT:SEP";
const NUMI = "STRUCT:NUMI";
const DENO = "STRUCT:DENO";
const FRAC = "NODE:FRAC";
const TERM = "NODE:TERM";

const OP_PREFIX = "OP:";
const OPVAR_PREFIX = "OPVAR:";
const VAR_PREFIX = "VAR:";
const COEF_PREFIX = "COEF:";
const EXP_PREFIX = "EXP:";

function serializeSlangMath(value) {
  const tokens = [];

  function serialize(node) {
    if (node == null) {
      throw new Error("Cannot serialize null or undefined slang node.");
    }
    if (Array.isArray(node)) {
      serializeTermList(node);
      return;
    }
    if (typeof node === "object") {
      if (typeof node.op === "string") {
        serializeOpNode(node);
        return;
      }
      if (node.numi !== undefined && node.deno !== undefined) {
        serializeFraction(node);
        return;
      }
      if (typeof node.coeff === "number") {
        serializeTerm(node);
        return;
      }
    }
    throw new Error(
      `Unsupported slang node type during serialization: ${JSON.stringify(node)}`,
    );
  }

  function serializeOpNode(node) {
    tokens.push(`${OP_PREFIX}${node.op}`);
    if (node.var !== undefined) {
      tokens.push(`${OPVAR_PREFIX}${node.var}`);
    }
    if (Array.isArray(node.vars)) {
      for (const variable of node.vars) {
        tokens.push(`${OPVAR_PREFIX}${variable}`);
      }
    }
    tokens.push(OPEN);

    const children = [];
    if (node.expr !== undefined) children.push(node.expr);
    if (node.u !== undefined) children.push(node.u);
    if (node.v !== undefined) children.push(node.v);
    if (node.left !== undefined) children.push(node.left);
    if (node.right !== undefined) children.push(node.right);
    if (Array.isArray(node.args)) children.push(...node.args);

    for (let i = 0; i < children.length; i += 1) {
      if (i > 0) tokens.push(SEP);
      serialize(children[i]);
    }
    tokens.push(CLOSE);
  }

  function serializeFraction(node) {
    tokens.push(FRAC, OPEN, NUMI, OPEN);
    serializeTermList(extractTerms(node.numi));
    tokens.push(CLOSE, SEP, DENO, OPEN);
    serializeTermList(extractTerms(node.deno));
    tokens.push(CLOSE, CLOSE);
  }

  function serializeTermList(terms) {
    if (!Array.isArray(terms)) {
      throw new Error(
        `Expected an array of terms, got ${JSON.stringify(terms)}`,
      );
    }
    if (terms.length === 0) {
      tokens.push(TERM, `${COEF_PREFIX}0`);
      return;
    }
    for (let i = 0; i < terms.length; i += 1) {
      if (i > 0) tokens.push(SEP);
      serialize(terms[i]);
    }
  }

  function serializeTerm(node) {
    tokens.push(TERM);
    if (typeof node.coeff !== "number") {
      throw new Error(
        `TERM node missing numeric coeff: ${JSON.stringify(node)}`,
      );
    }
    tokens.push(`${COEF_PREFIX}${node.coeff}`);
    if (node.var && typeof node.var === "object") {
      const varEntries = Object.entries(node.var).sort(([a], [b]) =>
        a.localeCompare(b),
      );
      for (const [name, exp] of varEntries) {
        tokens.push(`${VAR_PREFIX}${name}`);
        tokens.push(`${EXP_PREFIX}${exp}`);
      }
    }
  }

  function extractTerms(container) {
    if (container == null) {
      return [];
    }
    if (Array.isArray(container)) {
      return container;
    }
    if (container.terms !== undefined && Array.isArray(container.terms)) {
      return container.terms;
    }
    if (typeof container === "number") {
      return [{ coeff: container }];
    }
    throw new Error(
      `Unsupported fraction term container: ${JSON.stringify(container)}`,
    );
  }

  serialize(value);
  return tokens;
}

function deserializeSlangMath(tokens) {
  if (!Array.isArray(tokens)) {
    throw new Error("deserializeSlangMath expects an array of tokens.");
  }

  const { node, next } = parseNode(tokens, 0);
  if (next !== tokens.length) {
    throw new Error(
      `Extra tokens found after deserialization at position ${next}.`,
    );
  }
  return node;
}

function parseNode(tokens, index) {
  const token = tokens[index];
  if (token === TERM) {
    return parseTerm(tokens, index);
  }
  if (token === FRAC) {
    return parseFraction(tokens, index);
  }
  if (typeof token === "string" && token.startsWith(OP_PREFIX)) {
    return parseOpNode(tokens, index);
  }
  throw new Error(
    `Unexpected token while parsing node at index ${index}: ${token}`,
  );
}

function parseOpNode(tokens, index) {
  const opToken = tokens[index];
  const node = { op: opToken.slice(OP_PREFIX.length) };
  index += 1;

  while (
    index < tokens.length &&
    typeof tokens[index] === "string" &&
    tokens[index].startsWith(OPVAR_PREFIX)
  ) {
    const varName = tokens[index].slice(OPVAR_PREFIX.length);
    if (node.var === undefined) {
      node.var = varName;
    } else if (node.vars === undefined) {
      node.vars = [node.var, varName];
    } else {
      node.vars.push(varName);
    }
    index += 1;
  }

  index = expectToken(tokens, index, OPEN);
  const children = [];
  while (index < tokens.length && tokens[index] !== CLOSE) {
    const childResult = parseNode(tokens, index);
    children.push(childResult.node);
    index = childResult.next;
    if (tokens[index] === SEP) {
      index += 1;
    }
  }
  index = expectToken(tokens, index, CLOSE);

  if (children.length === 1) {
    node.expr = children[0];
  } else if (
    children.length === 2 &&
    ["product_rule", "quotient_rule"].includes(node.op)
  ) {
    node.u = children[0];
    node.v = children[1];
  } else if (children.length > 0) {
    node.children = children;
  }
  return { node, next: index };
}

function parseFraction(tokens, index) {
  index = expectToken(tokens, index, FRAC);
  index = expectToken(tokens, index, OPEN);
  index = expectToken(tokens, index, NUMI);
  const numeratorResult = parseWrappedTermList(tokens, index);
  index = numeratorResult.next;
  index = expectToken(tokens, index, SEP);
  index = expectToken(tokens, index, DENO);
  const denominatorResult = parseWrappedTermList(tokens, index);
  index = expectToken(tokens, denominatorResult.next, CLOSE);

  return {
    node: {
      numi: { terms: numeratorResult.terms },
      deno: { terms: denominatorResult.terms },
    },
    next: index,
  };
}

function parseWrappedTermList(tokens, index) {
  index = expectToken(tokens, index, OPEN);
  const terms = [];
  while (index < tokens.length && tokens[index] !== CLOSE) {
    const termResult = parseNode(tokens, index);
    terms.push(termResult.node);
    index = termResult.next;
    if (tokens[index] === SEP) {
      index += 1;
    }
  }
  index = expectToken(tokens, index, CLOSE);
  return { terms, next: index };
}

function parseTerm(tokens, index) {
  index = expectToken(tokens, index, TERM);
  const coefToken = tokens[index];
  if (typeof coefToken !== "string" || !coefToken.startsWith(COEF_PREFIX)) {
    throw new Error(
      `Expected COEF after TERM at index ${index}, got ${coefToken}`,
    );
  }
  const coeff = Number(coefToken.slice(COEF_PREFIX.length));
  if (Number.isNaN(coeff)) {
    throw new Error(`Invalid coefficient token: ${coefToken}`);
  }
  const node = { coeff };
  index += 1;

  while (
    index < tokens.length &&
    typeof tokens[index] === "string" &&
    tokens[index].startsWith(VAR_PREFIX)
  ) {
    const varName = tokens[index].slice(VAR_PREFIX.length);
    index += 1;
    const expToken = tokens[index];
    if (typeof expToken !== "string" || !expToken.startsWith(EXP_PREFIX)) {
      throw new Error(
        `Expected EXP token after VAR at index ${index}, got ${expToken}`,
      );
    }
    const exp = Number(expToken.slice(EXP_PREFIX.length));
    if (Number.isNaN(exp)) {
      throw new Error(`Invalid exponent token: ${expToken}`);
    }
    node.var = node.var || {};
    node.var[varName] = exp;
    index += 1;
  }

  return { node, next: index };
}

function expectToken(tokens, index, expected) {
  if (tokens[index] !== expected) {
    throw new Error(
      `Expected token ${expected} at index ${index}, got ${tokens[index]}`,
    );
  }
  return index + 1;
}

export {
  serializeSlangMath,
  deserializeSlangMath,
  OPEN,
  CLOSE,
  SEP,
  NUMI,
  DENO,
  FRAC,
  TERM,
};
