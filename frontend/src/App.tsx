import { MotionConfig } from 'motion/react'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/Shell'
import { EmptyState, LoadingBlock } from './components/ui'

const Overview = lazy(() => import('./pages/Overview'))
const Predictor = lazy(() => import('./pages/Predictor'))
const Performance = lazy(() => import('./pages/Performance'))
const Explain = lazy(() => import('./pages/Explain'))
const Explore = lazy(() => import('./pages/Explore'))
const About = lazy(() => import('./pages/About'))
const Researcher = lazy(() => import('./pages/Researcher'))
const Assistant = lazy(() => import('./pages/Assistant'))

function NotFound() {
  return <EmptyState title="Page not found" message="That route does not exist." action={<Link to="/" className="text-accent underline underline-offset-4">Back to Overview</Link>} />
}

export default function App() {
  return (
    // reducedMotion="user" makes every motion component honour prefers-reduced-motion
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <Suspense fallback={<div className="p-8"><LoadingBlock height="h-96" label="Loading page" /></div>}>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Overview />} />
              <Route path="predict" element={<Predictor />} />
              <Route path="performance" element={<Performance />} />
              <Route path="explain" element={<Explain />} />
              <Route path="explore" element={<Explore />} />
              <Route path="researcher" element={<Researcher />} />
              <Route path="assistant" element={<Assistant />} />
              <Route path="about" element={<About />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </MotionConfig>
  )
}
