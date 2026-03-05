# @oomfware/lang-detect

natural language detection library.

```sh
npm install @oomfware/lang-detect
```

## usage

call `initialize()` once to load model weights via `fetch()`, then use `detect()` synchronously on
any text. results are sorted by probability, highest first.

```ts
import { initialize, detect } from '@oomfware/lang-detect';

await initialize();

const results = detect('the quick brown fox jumps over the lazy dog');
console.log(results[0]); // -> ['eng', 0.98]
```

mixed-script text returns detections for each script present, with probabilities scaled by
proportion:

```ts
const results = detect('Hello Мир');
// -> [['eng', ...], ['rus', ...]]
```

### supported languages

50 languages across Latin, Cyrillic, Arabic, Devanagari, CJK, and unique-script families.

| code  | language         | code  | language         | code  | language   |
| ----- | ---------------- | ----- | ---------------- | ----- | ---------- |
| `afr` | Afrikaans        | `hau` | Hausa            | `por` | Portuguese |
| `ara` | Arabic           | `heb` | Hebrew           | `ron` | Romanian   |
| `aze` | Azerbaijani      | `hin` | Hindi            | `run` | Rundi      |
| `bel` | Belarusian       | `hrv` | Croatian         | `rus` | Russian    |
| `ben` | Bengali          | `hun` | Hungarian        | `slk` | Slovak     |
| `bul` | Bulgarian        | `hye` | Armenian         | `spa` | Spanish    |
| `cat` | Catalan          | `ind` | Indonesian       | `srp` | Serbian    |
| `ces` | Czech            | `isl` | Icelandic        | `swe` | Swedish    |
| `ckb` | Central Kurdish  | `ita` | Italian          | `tgl` | Tagalog    |
| `cmn` | Mandarin Chinese | `jpn` | Japanese         | `tur` | Turkish    |
| `dan` | Danish           | `kat` | Georgian         | `ukr` | Ukrainian  |
| `deu` | German           | `kaz` | Kazakh           | `vie` | Vietnamese |
| `ell` | Greek            | `kor` | Korean           |       |            |
| `eng` | English          | `lit` | Lithuanian       |       |            |
| `est` | Estonian         | `mar` | Marathi          |       |            |
| `eus` | Basque           | `mkd` | Macedonian       |       |            |
| `fin` | Finnish          | `nld` | Dutch            |       |            |
| `fra` | French           | `nob` | Norwegian Bokmål |       |            |
| `pes` | Persian          | `pol` | Polish           |       |            |
