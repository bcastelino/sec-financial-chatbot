export function RayBackground() {
  return (
    <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none select-none">
      <div className="absolute inset-0 bg-[#0f0f0f]" />
      <div
        className="absolute left-1/2 -translate-x-1/2 w-[4000px] h-[1800px] sm:w-[6000px]"
        style={{
          background:
            'radial-gradient(circle at center 800px, rgba(20, 136, 252, 0.8) 0%, rgba(20, 136, 252, 0.35) 14%, rgba(20, 136, 252, 0.18) 18%, rgba(20, 136, 252, 0.08) 22%, rgba(17, 17, 20, 0.2) 25%)',
        }}
      />
      <div
        className="absolute top-[175px] left-1/2 w-[1600px] h-[1600px] sm:top-1/2 sm:w-[3043px] sm:h-[2865px]"
        style={{ transform: 'translate(-50%) rotate(180deg)' }}
      >
        <div
          className="absolute w-full h-full rounded-full -mt-[13px]"
          style={{
            background: 'radial-gradient(43.89% 25.74% at 50.02% 97.24%, #111114 0%, #0f0f0f 100%)',
            border: '16px solid white',
            transform: 'rotate(180deg)',
            zIndex: 5,
          }}
        />
        <div
          className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[11px]"
          style={{ border: '23px solid #b7d7f6', transform: 'rotate(180deg)', zIndex: 4 }}
        />
        <div
          className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[8px]"
          style={{ border: '23px solid #8fc1f2', transform: 'rotate(180deg)', zIndex: 3 }}
        />
        <div
          className="absolute w-full h-full rounded-full bg-[#0f0f0f] -mt-[4px]"
          style={{ border: '23px solid #64acf6', transform: 'rotate(180deg)', zIndex: 2 }}
        />
        <div
          className="absolute w-full h-full rounded-full bg-[#0f0f0f]"
          style={{
            border: '20px solid #1172e2',
            boxShadow: '0 -15px 24.8px rgba(17, 114, 226, 0.6)',
            transform: 'rotate(180deg)',
            zIndex: 1,
          }}
        />
      </div>
    </div>
  )
}
