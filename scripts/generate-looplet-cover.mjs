import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const colors = {
  paper: '#fbfcfd', ink: '#23343b', muted: '#536772', line: '#c9d6dc',
  teal: '#08776a', tealWash: '#e7f4ee', blue: '#2c5f9f', blueWash: '#edf3fa',
  amber: '#906018', amberWash: '#fff5e5', red: '#b44331', redWash: '#fcefe9',
};

function escapeXml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;',
  })[character]);
}

function canvas(name, width, height) {
  const labels = [];
  const text = (x, y, value, options = {}) => {
    const { size = 24, weight = 400, tone = 'ink', anchor = 'start', mono = false,
      maxWidth = anchor === 'middle' ? Math.min(x, width - x) * 2 : width - x - 16 } = options;
    labels.push({ x, y, value, size, weight, anchor, mono, maxWidth });
    return `<text x="${x}" y="${y}" font-family="${mono ? 'DejaVu Sans Mono' : 'Lato'}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" fill="${colors[tone]}">${escapeXml(value)}</text>`;
  };
  const rect = (x, y, boxWidth, boxHeight, tone = 'line', fill = 'paper', dashed = false) =>
    `<rect x="${x}" y="${y}" width="${boxWidth}" height="${boxHeight}" rx="5" stroke="${colors[tone]}" stroke-width="1.6" fill="${colors[fill]}"${dashed ? ' stroke-dasharray="7 5"' : ''}/>`;
  const path = (data, tone = 'muted', { arrow = true, dashed = false } = {}) =>
    `<path d="${data}" fill="none" stroke="${colors[tone]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"${arrow ? ` marker-end="url(#${tone})"` : ''}${dashed ? ' stroke-dasharray="5 5"' : ''}/>`;
  const node = (x, y, boxWidth, boxHeight, title, options = {}) => {
    const { subtitle, tone = 'line', fill = 'paper', size = 25 } = options;
    return rect(x, y, boxWidth, boxHeight, tone, fill)
      + text(x + boxWidth / 2, y + (subtitle ? 32 : boxHeight / 2 + size * 0.34), title,
        { size, weight: 700, anchor: 'middle', maxWidth: boxWidth - 24 })
      + (subtitle ? text(x + boxWidth / 2, y + 61, subtitle,
        { size: 19, tone: 'muted', anchor: 'middle', maxWidth: boxWidth - 24 }) : '');
  };
  const heading = (number, title, subtitle, mobile = false) =>
    text(mobile ? 24 : 36, 25, `${number} / LOOPLET`, { size: 14, weight: 700, tone: 'teal' })
    + text(mobile ? 24 : 36, 64, title, { size: mobile ? 28 : 32, weight: 900 })
    + text(mobile ? 24 : 36, 96, subtitle, { size: 20, tone: 'muted' });
  const finish = content => ({
    name, width: width * 2, height: height * 2, labels,
    artwork: `<svg xmlns="http://www.w3.org/2000/svg" width="${width * 2}" height="${height * 2}" viewBox="0 0 ${width} ${height}">
      <defs>${['muted', 'teal', 'amber', 'blue', 'red'].map(tone => `<marker id="${tone}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 2 1 L 8 5 L 2 9" fill="none" stroke="${colors[tone]}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>`).join('')}
      <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 H 0 V 24" fill="none" stroke="#e7edf0" stroke-width="0.6"/></pattern></defs>
      <style>text { letter-spacing: 0; }</style>
      <rect width="${width}" height="${height}" fill="${colors.paper}"/>
      <rect width="${width}" height="${height}" fill="url(#grid)" opacity="0.55"/>
      <path d="M 0 0 H ${width}" stroke="${colors.teal}" stroke-width="7"/>
      ${content}</svg>`,
  });
  return { text, rect, path, node, heading, finish };
}

function controlLoop(mobile) {
  const { text, path, node, heading, finish } = canvas(
    `control-loop${mobile ? '-mobile' : ''}`, mobile ? 400 : 900, mobile ? 810 : 550,
  );
  if (mobile) {
    return finish([
      heading('02', 'Check before acting.', 'Check tools and requests to finish.', true),
      node(86, 142, 244, 80, 'Model', { subtitle: 'Choose the next step', size: 28, tone: 'blue', fill: 'blueWash' }),
      path('M 208 222 V 274', 'blue'),
      text(228, 256, 'tool request', { size: 20, tone: 'blue', maxWidth: 132 }),
      node(86, 286, 244, 80, 'Action check', { subtitle: 'Allow the call?', tone: 'amber', fill: 'amberWash', size: 25 }),
      path('M 208 366 V 418', 'teal'),
      text(228, 402, 'yes', { size: 20, tone: 'teal' }),
      node(86, 430, 244, 80, 'Tool', { subtitle: 'Run the operation', size: 28 }),
      path('M 330 182 H 376 V 630 H 342', 'blue'),
      text(364, 570, 'finish request', { size: 20, tone: 'blue', anchor: 'end', maxWidth: 154 }),
      node(86, 590, 244, 80, 'Completion check', { subtitle: 'Required work done?', tone: 'amber', fill: 'amberWash', size: 24 }),
      path('M 208 670 V 718', 'teal'),
      text(228, 704, 'yes', { size: 20, tone: 'teal' }),
      node(86, 730, 244, 54, 'Stop', { size: 27, tone: 'teal', fill: 'tealWash' }),
      path('M 86 326 H 24', 'amber', { arrow: false }),
      text(30, 311, 'deny', { size: 18, tone: 'amber', maxWidth: 50 }),
      path('M 86 470 H 24', 'teal', { arrow: false }),
      text(27, 455, 'result', { size: 18, tone: 'teal', maxWidth: 53 }),
      path('M 86 630 H 24', 'amber', { arrow: false }),
      text(30, 615, 'retry', { size: 18, tone: 'amber', maxWidth: 50 }),
      path('M 24 630 V 182 H 74'),
    ].join(''));
  }
  return finish([
    heading('02', 'Let the model choose. Keep control.', 'Check tool calls and requests to finish.'),
    node(36, 253, 210, 86, 'Model', { subtitle: 'Choose the next step', size: 28, tone: 'blue', fill: 'blueWash' }),
    path('M 246 296 H 302 V 221 H 362', 'blue'),
    text(330, 207, 'tool', { size: 20, tone: 'blue', anchor: 'middle', maxWidth: 64 }),
    node(374, 178, 216, 86, 'Action check', { subtitle: 'Allow the call?', tone: 'amber', fill: 'amberWash', size: 25 }),
    path('M 590 221 H 664', 'teal'),
    text(628, 207, 'yes', { size: 20, tone: 'teal', anchor: 'middle', maxWidth: 74 }),
    node(676, 178, 188, 86, 'Tool', { subtitle: 'Run the call', size: 28 }),
    path('M 482 178 V 131', 'amber', { arrow: false }),
    text(494, 160, 'denied', { size: 19, tone: 'amber', maxWidth: 94 }),
    path('M 770 178 V 131', 'teal', { arrow: false }),
    text(782, 160, 'result', { size: 19, tone: 'teal', maxWidth: 82 }),
    path('M 770 131 H 141 V 241'),
    path('M 302 296 V 430 H 362', 'blue'),
    text(330, 416, 'finish', { size: 19, tone: 'blue', anchor: 'middle', maxWidth: 70 }),
    node(374, 387, 216, 86, 'Completion check', { subtitle: 'Required work done?', tone: 'amber', fill: 'amberWash', size: 23 }),
    path('M 590 430 H 664', 'teal'),
    text(628, 416, 'yes', { size: 20, tone: 'teal', anchor: 'middle', maxWidth: 74 }),
    node(676, 387, 188, 86, 'Stop', { subtitle: 'Save stop reason', size: 28, tone: 'teal', fill: 'tealWash' }),
    path('M 482 473 V 525 H 141 V 351', 'amber'),
    text(282, 510, 'More work needed', { size: 21, tone: 'amber', maxWidth: 250 }),
  ].join(''));
}

async function cover(mobile) {
  const width = mobile ? 800 : 1800;
  const height = mobile ? 800 : 945;
  const palette = {
    paper: '#f7f9fc', ink: '#26343a', muted: '#68757f', rule: '#dce3eb',
    white: '#ffffff', blue: '#2853b6', paleBlue: '#e5ebf7',
    coral: '#d9523f', paleCoral: '#fbe7e1', green: '#147965', paleGreen: '#e4f2ec',
  };
  const layers = [];
  const block = (left, top, boxWidth, boxHeight, color) => {
    layers.push({
      input: { create: { width: boxWidth, height: boxHeight, channels: 4, background: color } },
      left, top,
    });
  };
  const label = async (left, top, value, size, color, options = {}) => {
    const { mono = false, bold = false, maxWidth = width - left - 30 } = options;
    const input = await sharp({ text: {
      text: `<span foreground="${color}">${escapeXml(value)}</span>`,
      font: `${mono ? 'DejaVu Sans Mono' : 'Lato'} ${bold ? 'Bold' : 'Regular'} ${size}`,
      dpi: 72, rgba: true,
    } }).png().toBuffer();
    const measured = await sharp(input).metadata();
    assert(measured.width <= maxWidth, `cover label too wide: ${value}`);
    assert(top + measured.height < height, `cover label too low: ${value}`);
    layers.push({ input, left, top });
  };
  const left = mobile ? 90 : 480;
  const top = mobile ? 142 : 176;
  const sheetWidth = mobile ? 620 : 840;
  const sheetHeight = mobile ? 440 : 498;
  for (let column = 24; column < width; column += 40) {
    for (let row = 24; row < height; row += 40) {
      block(column, row, 2, 2, palette.rule);
    }
  }
  block(left - 46, top - 46, sheetWidth, sheetHeight, palette.paleBlue);
  block(left - 23, top - 23, sheetWidth, sheetHeight, palette.blue);
  block(left + 12, top + 12, sheetWidth, sheetHeight, palette.rule);
  block(left, top, sheetWidth, sheetHeight, palette.white);
  block(left, top, 9, sheetHeight, palette.blue);
  block(left, top + 97, sheetWidth, 2, palette.rule);
  await label(mobile ? 46 : 100, mobile ? 40 : 55, 'LOOPLET', mobile ? 28 : 31, palette.blue, { bold: true });
  await label(left + 35, top + 33, 'agent.cartridge/', mobile ? 34 : 43, palette.ink, { mono: true, maxWidth: sheetWidth - 70 });
  const entries = ['prompts/', 'tools/', 'memory/', 'hooks/'];
  const lineHeight = mobile ? 65 : 76;
  for (const [index, entry] of entries.entries()) {
    const row = top + 125 + index * lineHeight;
    if (index === 3) {
      block(left + 25, row - 13, sheetWidth - 50, lineHeight - 3, palette.paleGreen);
    }
    await label(left + 42, row, entry, mobile ? 31 : 37, index === 3 ? palette.green : palette.muted, { mono: true });
    const marksLeft = left + (mobile ? 300 : 420);
    const marksWidth = mobile ? 246 : 326;
    block(marksLeft, row + 4, marksWidth - index * 22, 9, index === 3 ? palette.green : palette.paleBlue);
    block(marksLeft, row + 26, marksWidth - 60 - index * 16, 9, index === 3 ? palette.green : palette.paleBlue);
    if (index === 3) {
      block(left + sheetWidth - 47, row + 7, 4, 26, palette.green);
      block(left + sheetWidth - 58, row + 18, 26, 4, palette.green);
    }
  }
  const editLeft = mobile ? 39 : 334;
  const editTop = mobile ? 441 : 410;
  const editWidth = mobile ? 78 : 112;
  block(editLeft + 7, editTop + 7, editWidth, editWidth, palette.paleCoral);
  block(editLeft, editTop, editWidth, editWidth, palette.coral);
  block(editLeft + 21, editTop + Math.floor(editWidth / 2) - 3, editWidth - 42, 6, palette.white);
  block(editLeft + Math.floor(editWidth / 2) - 3, editTop + 21, 6, editWidth - 42, palette.white);
  const footerTop = mobile ? 662 : 790;
  const caption = 'Edit. Run. Evaluate.';
  const captionSize = mobile ? 45 : 66;
  const captionImage = await sharp({ text: {
    text: `<span foreground="${palette.ink}">${caption}</span>`,
    font: `Lato Bold ${captionSize}`, dpi: 72, rgba: true,
  } }).png().toBuffer();
  const captionMetadata = await sharp(captionImage).metadata();
  layers.push({ input: captionImage, left: Math.round((width - captionMetadata.width) / 2), top: footerTop });
  block(mobile ? 90 : 710, footerTop - 39, mobile ? 620 : 380, 3, palette.blue);
  const name = `platform-ownership${mobile ? '-mobile' : ''}`;
  const output = fileURLToPath(new URL(`../public/images/looplet/${name}.png`, import.meta.url));
  const result = await sharp({ create: { width, height, channels: 4, background: palette.paper } })
    .composite(layers).png({ compressionLevel: 9 }).toFile(output);
  assert.equal(result.width, width);
  assert.equal(result.height, height);
  const statistics = await sharp(output).stats();
  assert(statistics.channels.slice(0, 3).every(channel => channel.stdev > 20));
  console.log(`Generated ${name}.png: ${width} x ${height}, raster cover, ${Math.round(result.size / 1024)} KiB`);
}

function replay(mobile) {
  const { text, rect, path, heading, finish } = canvas(
    `replay-comparison${mobile ? '-mobile' : ''}`, mobile ? 400 : 900, mobile ? 760 : 580,
  );
  if (mobile) {
    return finish([
      heading('03', 'Check what happened.', 'Same request, different result.', true),
      rect(24, 134, 352, 72, 'blue', 'blueWash'),
      text(42, 165, 'Request $250, then finish', { size: 24, maxWidth: 316 }),
      text(42, 192, 'Fresh empty files in both runs', { size: 19, tone: 'muted', maxWidth: 316 }),
      path('M 24 171 H 10 V 298 H 30', 'blue'),
      path('M 10 298 V 493 H 30', 'blue'),
      text(42, 255, 'Prompt only', { size: 27, weight: 900 }),
      text(358, 255, 'FAIL', { size: 26, weight: 900, tone: 'red', anchor: 'end', maxWidth: 96 }),
      text(42, 305, '$250 recorded', { size: 31, weight: 700 }),
      text(42, 341, 'No review', { size: 25 }),
      text(42, 377, '0 tool errors', { size: 22, tone: 'muted' }),
      path('M 42 401 H 376', 'line', { arrow: false }),
      text(42, 447, 'Add the hook', { size: 27, weight: 900 }),
      text(358, 447, 'PASS', { size: 26, weight: 900, tone: 'teal', anchor: 'end', maxWidth: 96 }),
      text(42, 493, 'No refund', { size: 31, weight: 700 }),
      text(42, 530, 'One pending review', { size: 25 }),
      text(42, 566, '1 error: refund blocked', { size: 22, tone: 'amber' }),
      rect(24, 606, 352, 87),
      text(42, 638, 'Expected: no refund', { size: 25, weight: 700, maxWidth: 316 }),
      text(42, 671, '+ one pending review', { size: 25, weight: 700, maxWidth: 316 }),
      text(24, 735, 'Replay runs tools, not the model.', { size: 22, tone: 'muted' }),
    ].join(''));
  }
  return finish([
    heading('03', 'Same request. Different result.', 'Only the hook and its settings change.'),
    rect(36, 140, 828, 56, 'blue', 'blueWash'),
    text(450, 175, 'Replay: a $250 refund request, then finish.', { size: 25, anchor: 'middle' }),
    path('M 450 196 V 216 H 244 V 245', 'blue'),
    path('M 450 216 H 658 V 245', 'blue'),
    text(72, 284, 'Prompt only', { size: 28, weight: 900 }),
    text(502, 284, 'Add the hook', { size: 28, weight: 900 }),
    path('M 450 268 V 447', 'line', { arrow: false }),
    text(72, 332, '$250 recorded', { size: 33, weight: 700 }),
    text(502, 332, 'No refund', { size: 33, weight: 700 }),
    text(72, 370, 'No review', { size: 26 }),
    text(502, 370, 'One pending review', { size: 26 }),
    text(72, 408, '0 tool errors', { size: 23, tone: 'muted' }),
    text(502, 408, '1 error: refund blocked', { size: 23, tone: 'amber' }),
    text(72, 451, 'FAIL', { size: 30, weight: 900, tone: 'red' }),
    text(502, 451, 'PASS', { size: 30, weight: 900, tone: 'teal' }),
    rect(36, 480, 828, 76),
    text(58, 511, 'Expected: no refund and one pending review.', { size: 27, weight: 700 }),
    text(58, 543, 'Empty files. Same evals. No new model call.', { size: 22, tone: 'muted' }),
  ].join(''));
}

function improvement(mobile) {
  const { text, path, node, heading, finish } = canvas(
    `improvement-loop${mobile ? '-mobile' : ''}`, mobile ? 400 : 900, mobile ? 740 : 500,
  );
  if (mobile) {
    return finish([
      heading('04', 'Build the next version.', 'Run, evaluate, revise.', true),
      node(64, 140, 272, 70, 'Builder', { subtitle: 'Edit the definition', size: 26, tone: 'blue', fill: 'blueWash' }),
      path('M 200 210 V 232', 'blue'),
      node(64, 244, 272, 70, 'Candidate', { subtitle: 'Edited cartridge', size: 26, tone: 'blue', fill: 'blueWash' }),
      path('M 200 314 V 336'),
      node(64, 348, 272, 70, 'Run', { subtitle: 'Isolated test workspace', size: 26, tone: 'amber', fill: 'amberWash' }),
      path('M 200 418 V 472'),
      path('M 52 443 H 376', 'line', { arrow: false, dashed: true }),
      text(218, 464, 'HOST CONTROLS', { size: 14, weight: 700, tone: 'teal', maxWidth: 158 }),
      node(64, 484, 272, 70, 'Check results', { subtitle: 'Outside agent control', size: 25, tone: 'teal', fill: 'tealWash' }),
      path('M 64 519 H 24 V 175 H 52', 'amber'),
      path('M 200 554 V 608', 'teal'),
      text(220, 589, 'pass', { size: 21, tone: 'teal' }),
      node(64, 620, 272, 76, 'Release rules', { subtitle: 'Deploy or request review', size: 25, tone: 'teal', fill: 'tealWash' }),
      text(24, 726, 'Feedback from development tests', { size: 21, tone: 'amber', maxWidth: 352 }),
    ].join(''));
  }
  return finish([
    heading('04', 'Edit the agent, then test it.', 'The builder uses development results but cannot change the final checks.'),
    text(36, 155, 'EDITABLE FILES', { size: 15, weight: 700, tone: 'blue' }),
    text(504, 155, 'ISOLATED RUN', { size: 15, weight: 700, tone: 'amber' }),
    text(712, 155, 'HOST CONTROLS', { size: 15, weight: 700, tone: 'teal' }),
    path('M 696 134 V 477', 'line', { arrow: false, dashed: true }),
    node(36, 182, 170, 80, 'Builder', { subtitle: 'Edit files', size: 26, tone: 'blue', fill: 'blueWash' }),
    path('M 206 222 H 258', 'blue'),
    node(270, 182, 170, 80, 'Candidate', { subtitle: 'New cartridge', size: 26, tone: 'blue', fill: 'blueWash' }),
    path('M 440 222 H 492'),
    node(504, 182, 158, 80, 'Run', { subtitle: 'Test tasks', size: 26, tone: 'amber', fill: 'amberWash' }),
    path('M 662 222 H 716'),
    node(728, 182, 136, 80, 'Evaluate', { subtitle: 'Outputs', size: 23, tone: 'teal', fill: 'tealWash' }),
    path('M 760 262 V 331 H 121 V 274', 'amber'),
    text(370, 315, 'Development test results', { size: 22, tone: 'amber', anchor: 'middle', maxWidth: 490 }),
    path('M 830 262 V 384', 'teal'),
    text(814, 358, 'pass', { size: 19, tone: 'teal', anchor: 'end', maxWidth: 52 }),
    node(712, 396, 156, 72, 'Release rules', { subtitle: 'Deploy or wait', size: 20, tone: 'teal', fill: 'tealWash' }),
    text(36, 413, 'The builder and candidate cannot change', { size: 23, maxWidth: 600 }),
    text(36, 449, 'the final checks or their own permissions.', { size: 23, maxWidth: 600 }),
  ].join(''));
}

const figures = [
  controlLoop(false), controlLoop(true),
  replay(false), replay(true), improvement(false), improvement(true),
];
const measurements = new Map();
const layoutErrors = [];

for (const figure of figures) {
  for (const label of figure.labels) {
    const key = JSON.stringify([label.value, label.size, label.weight, label.mono]);
    if (!measurements.has(key)) {
      measurements.set(key, await sharp({ text: {
        text: escapeXml(label.value),
        font: `${label.mono ? 'DejaVu Sans Mono' : 'Lato'} ${label.weight >= 700 ? 'Bold' : 'Regular'} ${label.size}`,
        dpi: 72,
      } }).metadata());
    }
    const measured = measurements.get(key);
    if (measured.width > label.maxWidth) {
      layoutErrors.push(`${figure.name}: label too wide: ${label.value} (${measured.width} > ${label.maxWidth})`);
    }
    assert(label.y < figure.height / 2 && label.y - measured.height >= 0, `${figure.name}: label outside vertical bounds: ${label.value}`);
  }
  const output = fileURLToPath(new URL(`../public/images/looplet/${figure.name}.png`, import.meta.url));
  const result = await sharp(Buffer.from(figure.artwork))
    .png({ compressionLevel: 9 })
    .toFile(output);
  assert.equal(result.width, figure.width);
  assert.equal(result.height, figure.height);
  const statistics = await sharp(output).stats();
  assert(statistics.channels.slice(0, 3).every(channel => channel.stdev > 20));
  console.log(`Generated ${figure.name}.png: ${figure.width} x ${figure.height}, ${figure.labels.length} labels checked, ${Math.round(result.size / 1024)} KiB`);
}

assert.equal(layoutErrors.length, 0, layoutErrors.join('\n'));

await cover(false);
await cover(true);