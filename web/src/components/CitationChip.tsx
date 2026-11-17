interface Props {
  number: number
  onOpen: (number: number, trigger: HTMLElement) => void
}

/** A citation marker such as [2] in an answer, shown as a small chip that opens that source. */
export default function CitationChip({ number, onOpen }: Props) {
  return (
    <button
      type="button"
      className="cite"
      aria-label={`Source ${number}`}
      title={`Show source ${number}`}
      onClick={(event) => onOpen(number, event.currentTarget)}
    >
      {number}
    </button>
  )
}
